terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "=2.74.0"
    }
    azuread = {
      source  = "hashicorp/azuread"
      version = "=2.7.0"
    }
    http = {
      source = "hashicorp/http"
      version = "2.1.0"
    }
    tls = {
      version = "3.1.0"
    }
  }
  backend "azurerm" {}
}

provider "azurerm" {
  features {}
}

locals {
  acr_name = var.acr_name == "" ? var.az_resource_group_name : var.acr_name
  # An IP toward the top of the batch-worker-subnet IP address range
  # that we will use for the bath-worker side of the internal-gateway
  internal_ip = "10.128.255.254"
}

data "azurerm_resource_group" "rg" {
  name = var.az_resource_group_name
}

resource "azurerm_virtual_network" "default" {
  name                = "default"
  resource_group_name = data.azurerm_resource_group.rg.name
  address_space       = ["10.0.0.0/8"]
  location            = data.azurerm_resource_group.rg.location
}

resource "azurerm_subnet" "k8s_subnet" {
  name                 = "k8s-subnet"
  address_prefixes     = ["10.240.0.0/16"]
  resource_group_name  = data.azurerm_resource_group.rg.name
  virtual_network_name = azurerm_virtual_network.default.name

  enforce_private_link_endpoint_network_policies = true
}

resource "azurerm_subnet" "batch_worker_subnet" {
  name                 = "batch-worker-subnet"
  address_prefixes     = ["10.128.0.0/16"]
  resource_group_name  = data.azurerm_resource_group.rg.name
  virtual_network_name = azurerm_virtual_network.default.name

  enforce_private_link_endpoint_network_policies = true
}

resource "azurerm_kubernetes_cluster" "vdc" {
  name                = "vdc"
  resource_group_name = data.azurerm_resource_group.rg.name
  location            = data.azurerm_resource_group.rg.location
  dns_prefix          = "example"

  default_node_pool {
    name           = "default"
    node_count     = 1
    vm_size        = var.k8s_machine_type
    vnet_subnet_id = azurerm_subnet.k8s_subnet.id
    type           = "VirtualMachineScaleSets"
  }

  identity {
    type = "SystemAssigned"
  }
}

resource "azurerm_kubernetes_cluster_node_pool" "vdc_noonpreemptible_pool" {
  name                  = "nonpreempt"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.vdc.id
  vm_size               = var.k8s_machine_type
  vnet_subnet_id        = azurerm_subnet.k8s_subnet.id

  enable_auto_scaling = true

  min_count = 0
  max_count = 200

  node_labels = {
    "preemptible" = "false"
  }
}

resource "azurerm_kubernetes_cluster_node_pool" "vdc_preemptible_pool" {
  name                  = "preempt"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.vdc.id
  vm_size               = var.k8s_machine_type
  vnet_subnet_id        = azurerm_subnet.k8s_subnet.id

  enable_auto_scaling = true

  min_count = 0
  max_count = 200

  priority        = "Spot"
  eviction_policy = "Delete"
  node_labels = {
    "preemptible"                           = "true"
    "kubernetes.azure.com/scalesetpriority" = "spot"
  }
  node_taints = [
    "kubernetes.azure.com/scalesetpriority=spot:NoSchedule"
  ]
}

resource "azurerm_public_ip" "gateway_ip" {
  name                = "gateway-ip"
  resource_group_name = azurerm_kubernetes_cluster.vdc.node_resource_group
  location            = data.azurerm_resource_group.rg.location
  sku                 = "Standard"
  allocation_method   = "Static"
}

resource "azurerm_container_registry" "acr" {
  name                = local.acr_name
  resource_group_name = data.azurerm_resource_group.rg.name
  location            = data.azurerm_resource_group.rg.location
  sku                 = var.acr_sku
  admin_enabled       = true
}

resource "azurerm_role_assignment" "vdc_to_acr" {
  scope                = azurerm_container_registry.acr.id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_kubernetes_cluster.vdc.kubelet_identity[0].object_id
}

resource "azurerm_role_assignment" "vdc_batch_worker_subnet" {
  scope                = azurerm_subnet.batch_worker_subnet.id
  role_definition_name = "Network Contributor"
  principal_id         = azurerm_kubernetes_cluster.vdc.identity[0].principal_id
}

resource "azurerm_role_assignment" "vdc_k8s_subnet" {
  scope                = azurerm_subnet.k8s_subnet.id
  role_definition_name = "Network Contributor"
  principal_id         = azurerm_kubernetes_cluster.vdc.identity[0].principal_id
}

resource "random_id" "db_name_suffix" {
  byte_length = 4
}

resource "random_password" "db_root_password" {
  length = 22
}

resource "azurerm_mysql_server" "db" {
  name                = "db-${random_id.db_name_suffix.hex}"
  resource_group_name = data.azurerm_resource_group.rg.name
  location            = data.azurerm_resource_group.rg.location

  administrator_login          = "mysqladmin"
  administrator_login_password = random_password.db_root_password.result

  version    = "5.7"
  sku_name   = "GP_Gen5_2" # 2 vCPU, 10GiB
  storage_mb = 5120

  ssl_enforcement_enabled       = true
  public_network_access_enabled = false
}

data "http" "db_ca_cert" {
  url = "https://www.digicert.com/CACerts/BaltimoreCyberTrustRoot.crt.pem"
}

resource "azurerm_private_endpoint" "db_k8s_endpoint" {
  name                = "${azurerm_mysql_server.db.name}-k8s-endpoint"
  resource_group_name = data.azurerm_resource_group.rg.name
  location            = data.azurerm_resource_group.rg.location
  subnet_id           = azurerm_subnet.k8s_subnet.id

  private_service_connection {
    name                           = "${azurerm_mysql_server.db.name}-k8s-endpoint"
    private_connection_resource_id = azurerm_mysql_server.db.id
    subresource_names              = [ "mysqlServer" ]
    is_manual_connection           = false
  }
}

resource "tls_private_key" "db_client_key" {
  algorithm = "RSA"
}

resource "tls_self_signed_cert" "db_client_cert" {
  key_algorithm   = tls_private_key.db_client_key.algorithm
  private_key_pem = tls_private_key.db_client_key.private_key_pem

  subject {
    common_name  = "hail-client"
  }

  validity_period_hours = 24 * 365

  allowed_uses = [
    "client_auth"
  ]
}

resource "azurerm_user_assigned_identity" "batch_worker" {
  name                = "batch-worker"
  resource_group_name = data.azurerm_resource_group.rg.name
  location            = data.azurerm_resource_group.rg.location
}

resource "azurerm_role_assignment" "batch_worker" {
  scope                = data.azurerm_resource_group.rg.id
  role_definition_name = "acrpull"
  principal_id         = azurerm_user_assigned_identity.batch_worker.principal_id
}

resource "azurerm_shared_image_gallery" "batch" {
  name                = "${data.azurerm_resource_group.rg.name}_batch"
  resource_group_name = data.azurerm_resource_group.rg.name
  location            = data.azurerm_resource_group.rg.location
}

resource "azurerm_shared_image" "batch_worker" {
  name                = "batch-worker"
  gallery_name        = azurerm_shared_image_gallery.batch.name
  resource_group_name = data.azurerm_resource_group.rg.name
  location            = data.azurerm_resource_group.rg.location
  os_type             = "Linux"
  specialized	        = false

  identifier {
    publisher = "Canonical"
    offer     = "UbuntuServer"
    sku       = "20.04-LTS"
  }
}

provider "azuread" {}

resource "azuread_application" "auth" {
  display_name = "${data.azurerm_resource_group.rg.name}-auth"

  required_resource_access {
    resource_app_id = "00000003-0000-0000-c000-000000000000"

    resource_access {
      # Application.ReadWrite.All
      id   = "1bfefb4e-e0b5-418b-a88f-73c46d2cc8e9"
      type = "Role"
    }
  }
}
module "auth_sp" {
  source = "./service_principal"
  application_id = azuread_application.auth.application_id
  object_id      = azuread_application.auth.object_id
}

resource "azuread_application" "benchmark" {
  display_name = "${data.azurerm_resource_group.rg.name}-benchmark"
}
module "benchmark_sp" {
  source = "./service_principal"
  application_id = azuread_application.benchmark.application_id
  object_id      = azuread_application.benchmark.object_id
}

resource "azuread_application" "ci" {
  display_name = "${data.azurerm_resource_group.rg.name}-ci"
}
module "ci_sp" {
  source = "./service_principal"
  application_id = azuread_application.ci.application_id
  object_id      = azuread_application.ci.object_id
}

resource "azuread_application" "test" {
  display_name = "${data.azurerm_resource_group.rg.name}-test"
}
module "test_sp" {
  source = "./service_principal"
  application_id = azuread_application.test.application_id
  object_id      = azuread_application.test.object_id
}

resource "azuread_application" "test_dev" {
  display_name = "${data.azurerm_resource_group.rg.name}-test-dev"
}
module "test_dev_sp" {
  source = "./service_principal"
  application_id = azuread_application.test_dev.application_id
  object_id      = azuread_application.test_dev.object_id
}

resource "azuread_application" "query" {
  display_name = "${data.azurerm_resource_group.rg.name}-query"
}
module "query_sp" {
  source = "./service_principal"
  application_id = azuread_application.query.application_id
  object_id      = azuread_application.query.object_id
}

resource "azuread_application" "grafana" {
  display_name = "${data.azurerm_resource_group.rg.name}-grafana"
}
module "grafana_sp" {
  source = "./service_principal"
  application_id = azuread_application.grafana.application_id
  object_id      = azuread_application.grafana.object_id
}
