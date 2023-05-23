resource "azurerm_virtual_network" "default" {
  name                = "default"
  resource_group_name = var.resource_group.name
  address_space       = ["10.0.0.0/8"]
  location            = var.resource_group.location
}

resource "azurerm_subnet" "k8s_subnet" {
  name                 = "k8s-subnet"
  address_prefixes     = ["10.240.0.0/16"]
  resource_group_name  = var.resource_group.name
  virtual_network_name = azurerm_virtual_network.default.name
}

resource "azurerm_subnet" "batch_worker_subnet" {
  name                 = "batch-worker-subnet"
  address_prefixes     = ["10.128.0.0/16"]
  resource_group_name  = var.resource_group.name
  virtual_network_name = azurerm_virtual_network.default.name
}

resource "azurerm_subnet" "db_subnet" {
  name                 = "db-subnet"
  address_prefixes     = ["10.44.0.0/24"]
  resource_group_name  = var.resource_group.name
  virtual_network_name = azurerm_virtual_network.default.name

  service_endpoints = ["Microsoft.Storage"]
  delegation {
    name = "mysql-flexible-sever"
    service_delegation {
      name    = "Microsoft.DBforMySQL/flexibleServers"
      actions = [
        "Microsoft.Network/virtualNetworks/subnets/join/action",
      ]
    }
  }
}

resource "azurerm_log_analytics_workspace" "logs" {
  name                = "${var.resource_group.name}-logs"
  location            = var.resource_group.location
  resource_group_name = var.resource_group.name
  retention_in_days   = 30
}

resource "azurerm_kubernetes_cluster" "vdc" {
  name                = "vdc"
  resource_group_name = var.resource_group.name
  location            = var.resource_group.location
  dns_prefix          = "example"

  default_node_pool {
    name           = "nonpreempt"
    vm_size        = var.k8s_default_node_pool_machine_type
    vnet_subnet_id = azurerm_subnet.k8s_subnet.id

    enable_auto_scaling = true

    min_count = 1
    max_count = 5

    node_labels = {
      "preemptible" = "false"
    }
  }

  identity {
    type = "SystemAssigned"
  }

  addon_profile {
    oms_agent {
      enabled = true
      log_analytics_workspace_id = azurerm_log_analytics_workspace.logs.id
    }
  }

  # https://github.com/hashicorp/terraform-provider-azurerm/issues/7396
  lifecycle {
    ignore_changes = [addon_profile.0, default_node_pool.0.node_count]
  }
}

resource "azurerm_kubernetes_cluster_node_pool" "vdc_nonpreemptible_pool" {
  name                  = var.k8s_nonpreemptible_node_pool_name
  kubernetes_cluster_id = azurerm_kubernetes_cluster.vdc.id
  vm_size               = var.k8s_user_pool_machine_type
  vnet_subnet_id        = azurerm_subnet.k8s_subnet.id

  enable_auto_scaling = true

  min_count = 0
  max_count = 200

  node_labels = {
    "preemptible" = "false"
  }

  lifecycle {
    # Ignore if the node count has natually changed since last apply
    # due to autoscaling
    ignore_changes = [node_count]
  }
}

resource "azurerm_kubernetes_cluster_node_pool" "vdc_preemptible_pool" {
  name                  = var.k8s_preemptible_node_pool_name
  kubernetes_cluster_id = azurerm_kubernetes_cluster.vdc.id
  vm_size               = var.k8s_user_pool_machine_type
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

  lifecycle {
    # Ignore if the node count has natually changed since last apply
    # due to autoscaling
    ignore_changes = [node_count]
  }
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

resource "azurerm_role_assignment" "vdc_to_acr" {
  scope                = var.container_registry_id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_kubernetes_cluster.vdc.kubelet_identity[0].object_id
}

resource "azurerm_public_ip" "gateway_ip" {
  name                = "gateway-ip"
  resource_group_name = azurerm_kubernetes_cluster.vdc.node_resource_group
  location            = var.resource_group.location
  sku                 = "Standard"
  allocation_method   = "Static"
}

resource "kubernetes_pod_disruption_budget" "kube_dns_pdb" {
  metadata {
    name = "kube-dns"
    namespace = "kube-system"
  }
  spec {
    max_unavailable = "1"
    selector {
      match_labels = {
        k8s-app = "kube-dns"
      }
    }
  }
}
