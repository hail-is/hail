terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "=2.74.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "1.13.3"
    }
  }
  backend "azurerm" {}
}

provider "azurerm" {
  features {}
}

locals {
  acr_name = var.acr_name == "" ? var.az_resource_group_name : var.acr_name
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
}

resource "azurerm_kubernetes_cluster" "vdc" {
  name                = "vdc"
  resource_group_name = data.azurerm_resource_group.rg.name
  location            = data.azurerm_resource_group.rg.location
  dns_prefix          = "example"

  default_node_pool {
    name           = "default"
    node_count     = 1
    vm_size        = "Standard_D2_v2"
    vnet_subnet_id = azurerm_subnet.k8s_subnet.id
  }

  identity {
    type = "SystemAssigned"
  }
}

resource "azurerm_kubernetes_cluster_node_pool" "vdc_pool" {
  name                  = "pool"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.vdc.id
  vm_size               = "Standard_D2_v2"
  vnet_subnet_id        = azurerm_subnet.k8s_subnet.id

  enable_auto_scaling = true

  min_count = 0
  max_count = 200
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

resource "azurerm_private_endpoint" "db_endpoint" {
  name                = "${azurerm_mysql_server.db.name}-endpoint"
  resource_group_name = data.azurerm_resource_group.rg.name
  location            = data.azurerm_resource_group.rg.location
  subnet_id           = azurerm_subnet.k8s_subnet.id

  private_service_connection {
    name                           = "${azurerm_mysql_server.db.name}-endpoint"
    private_connection_resource_id = azurerm_mysql_server.db.id
    subresource_names              = [ "mysqlServer" ]
    is_manual_connection           = false
  }
}
