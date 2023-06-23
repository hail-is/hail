data "azurerm_subscription" "primary" {}

resource "azurerm_network_security_group" "batch_worker" {
  name                = "batch-worker-nsg"
  location            = var.resource_group.location
  resource_group_name = var.resource_group.name

  security_rule {
    name                       = "default-allow-ssh"
    priority                   = 1000
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "22"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }
}

resource "azurerm_shared_image_gallery" "batch" {
  name                = "${var.resource_group.name}_batch"
  resource_group_name = var.resource_group.name
  location            = var.resource_group.location
}

resource "azurerm_shared_image" "batch_worker" {
  name                = "batch-worker"
  gallery_name        = azurerm_shared_image_gallery.batch.name
  resource_group_name = var.resource_group.name
  location            = var.resource_group.location
  os_type             = "Linux"
  specialized	        = false

  identifier {
    publisher = "Canonical"
    offer     = "UbuntuServer"
    sku       = "20.04-LTS"
  }
}

resource "kubernetes_secret" "batch_worker_ssh_public_key" {
  metadata {
    name = "batch-worker-ssh-public-key"
  }

  data = {
    "ssh_rsa.pub" = file("~/.ssh/batch_worker_ssh_rsa.pub")
  }
}

resource "azurerm_storage_account" "batch" {
  name                     = "${var.resource_group.name}batch${var.storage_account_suffix}"
  resource_group_name      = var.resource_group.name
  location                 = var.resource_group.location
  account_tier             = "Standard"
  account_replication_type = "LRS"

  blob_properties {
    last_access_time_enabled = true
  }
}

resource "azurerm_storage_container" "batch_logs" {
  name                  = "logs"
  storage_account_name  = azurerm_storage_account.batch.name
  container_access_type = "private"
}

resource "azurerm_storage_container" "query" {
  name                  = "query"
  storage_account_name  = azurerm_storage_account.batch.name
  container_access_type = "private"
}

resource "azurerm_storage_account" "test" {
  name                     = "${var.batch_test_user_storage_account_name}test${var.storage_account_suffix}"
  resource_group_name      = var.resource_group.name
  location                 = var.resource_group.location
  account_tier             = "Standard"
  account_replication_type = "LRS"

  blob_properties {
    last_access_time_enabled = true
  }
}

resource "azurerm_storage_container" "test" {
  name                  = "test"
  storage_account_name  = azurerm_storage_account.test.name
  container_access_type = "private"
}

resource "azurerm_storage_management_policy" "test" {
  storage_account_id = azurerm_storage_account.test.id

  rule {
    name    = "test-artifacts-retention-1-day"
    enabled = true
    filters {
      prefix_match = [azurerm_storage_container.test.name]
      blob_types   = ["blockBlob"]
    }
    actions {
      base_blob {
        delete_after_days_since_modification_greater_than = 1
      }
    }
  }
}

resource "azurerm_user_assigned_identity" "batch_worker" {
  name                = "batch-worker"
  resource_group_name = var.resource_group.name
  location            = var.resource_group.location
}

resource "azurerm_role_assignment" "batch_worker" {
  scope                = var.container_registry_id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_user_assigned_identity.batch_worker.principal_id
}

resource "azurerm_role_assignment" "batch_worker_batch_account_contributor" {
  scope                = azurerm_storage_account.batch.id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = azurerm_user_assigned_identity.batch_worker.principal_id
}

resource "azurerm_role_assignment" "batch_worker_virtual_machine_contributor" {
  scope                = data.azurerm_subscription.primary.id
  role_definition_name = "Virtual Machine Contributor"
  principal_id         = azurerm_user_assigned_identity.batch_worker.principal_id
}

resource "azurerm_role_assignment" "batch_worker_test_container_contributor" {
  scope                = azurerm_storage_container.test.resource_manager_id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = azurerm_user_assigned_identity.batch_worker.principal_id
}

resource "azuread_application" "batch" {
  display_name = "${var.resource_group.name}-batch"
}
module "batch_sp" {
  source = "../service_principal"

  name                  = "batch"
  application_id        = azuread_application.batch.application_id
  application_object_id = azuread_application.batch.object_id

  subscription_resource_id = data.azurerm_subscription.primary.id
  subscription_roles = [
    "Reader",
    "Virtual Machine Contributor",
  ]

  resource_group_id = var.resource_group.id
  resource_group_roles = [
    "Network Contributor",
    "Managed Identity Operator",
    "Log Analytics Contributor",
  ]
}

resource "azurerm_role_assignment" "batch_batch_account_contributor" {
  scope                = azurerm_storage_account.batch.id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = module.batch_sp.principal_id
}

resource "azurerm_role_assignment" "batch_test_container_contributor" {
  scope                = azurerm_storage_container.test.resource_manager_id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = module.batch_sp.principal_id
}

resource "azuread_application" "test_batch" {
  display_name = "${var.resource_group.name}-test-batch"
}
module "test_batch_sp" {
  source = "../service_principal"

  name                  = "test-batch"
  application_id        = azuread_application.test_batch.application_id
  application_object_id = azuread_application.test_batch.object_id

  subscription_resource_id = data.azurerm_subscription.primary.id
  subscription_roles = [
    "Reader",
    "Virtual Machine Contributor",
  ]

  resource_group_id = var.resource_group.id
  resource_group_roles = [
    "Network Contributor",
    "Managed Identity Operator",
    "Log Analytics Contributor",
  ]
}

resource "azurerm_role_assignment" "test_batch_test_container_contributor" {
  scope                = azurerm_storage_container.test.resource_manager_id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = module.test_batch_sp.principal_id
}

# FIXME Now that there are test identities for each service, the test user no longer
# needs this many permissions. Perform an audit to see which can be removed
resource "azuread_application" "test" {
  display_name = "${var.resource_group.name}-test"

  required_resource_access {
    resource_app_id = "00000003-0000-0000-c000-000000000000"

    resource_access {
      # Application.ReadWrite.All
      id   = "1bfefb4e-e0b5-418b-a88f-73c46d2cc8e9"
      type = "Role"
    }
  }
}
module "test_sp" {
  source = "../service_principal"

  name                  = "test"
  application_id        = azuread_application.test.application_id
  application_object_id = azuread_application.test.object_id

  subscription_resource_id = data.azurerm_subscription.primary.id
  subscription_roles = [
    "Reader",
    "Virtual Machine Contributor",
  ]

  resource_group_id = var.resource_group.id
  resource_group_roles = [
    "Network Contributor",
    "Managed Identity Operator",
    "Log Analytics Contributor"
  ]
}

resource "azurerm_role_assignment" "test_test_container_contributor" {
  scope                = azurerm_storage_container.test.resource_manager_id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = module.test_sp.principal_id
}

resource "azurerm_role_assignment" "test_registry_viewer" {
  scope                = var.container_registry_id
  role_definition_name = "AcrPull"
  principal_id         = module.test_sp.principal_id
}

resource "azuread_application" "test_test" {
  display_name = "${var.resource_group.name}-test-test"
}
module "test_test_sp" {
  source = "../service_principal"

  name                  = "test-test"
  application_id        = azuread_application.test_test.application_id
  application_object_id = azuread_application.test_test.object_id
}

resource "azurerm_role_assignment" "test_test_sp_test_container_contributor" {
  scope                = azurerm_storage_container.test.resource_manager_id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = module.test_test_sp.principal_id
}

# Necessary to generate SAS tokens
resource "azurerm_role_assignment" "test_test_sp_test_account_key_operator" {
  scope                = azurerm_storage_account.test.id
  role_definition_name = "Storage Account Key Operator Service Role"
  principal_id         = module.test_test_sp.principal_id
}

resource "azurerm_role_assignment" "test_test_registry_viewer" {
  scope                = var.container_registry_id
  role_definition_name = "AcrPull"
  principal_id         = module.test_test_sp.principal_id
}

resource "azuread_application" "ci" {
  display_name = "${var.resource_group.name}-ci"
}
module "ci_sp" {
  source = "../service_principal"

  name                  = "ci"
  application_id        = azuread_application.ci.application_id
  application_object_id = azuread_application.ci.object_id
}

resource "azurerm_role_assignment" "ci_acr_role" {
  for_each = toset(["AcrPush", "AcrDelete"])

  scope                = var.container_registry_id
  role_definition_name = each.key
  principal_id         = module.ci_sp.principal_id
}

resource "kubernetes_secret" "registry_push_credentials" {
  metadata {
    name = "registry-push-credentials"
  }

  data = {
    "credentials.json" = jsonencode(module.ci_sp.credentials)
  }
}

resource "azuread_application" "test_ci" {
  display_name = "${var.resource_group.name}-test-ci"
}
module "test_ci_sp" {
  source = "../service_principal"

  name                  = "test-ci"
  application_id        = azuread_application.test_ci.application_id
  application_object_id = azuread_application.test_ci.object_id
}

resource "azurerm_role_assignment" "test_ci_test_container_contributor" {
  scope                = azurerm_storage_container.test.resource_manager_id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = module.test_ci_sp.principal_id
}

resource "azurerm_role_assignment" "test_ci_acr_role" {
  for_each = toset(["AcrPush", "AcrDelete"])

  scope                = var.container_registry_id
  role_definition_name = each.key
  principal_id         = module.test_ci_sp.principal_id
}

resource "azuread_application" "test_dev" {
  display_name = "${var.resource_group.name}-test-dev"
}
module "test_dev_sp" {
  source = "../service_principal"

  name                  = "test-dev"
  application_id        = azuread_application.test_dev.application_id
  application_object_id = azuread_application.test_dev.object_id
}

resource "azurerm_role_assignment" "test_dev_registry_viewer" {
  scope                = var.container_registry_id
  role_definition_name = "AcrPull"
  principal_id         = module.test_dev_sp.principal_id
}

resource "azuread_application" "test_test_dev" {
  display_name = "${var.resource_group.name}-test-test-dev"
}
module "test_test_dev_sp" {
  source = "../service_principal"

  name                  = "test-test-dev"
  application_id        = azuread_application.test_test_dev.application_id
  application_object_id = azuread_application.test_test_dev.object_id
}

resource "azurerm_role_assignment" "test_test_dev_registry_viewer" {
  scope                = var.container_registry_id
  role_definition_name = "AcrPull"
  principal_id         = module.test_test_dev_sp.principal_id
}

resource "azuread_application" "grafana" {
  display_name = "${var.resource_group.name}-grafana"
}
module "grafana_sp" {
  source = "../service_principal"

  name                  = "grafana"
  application_id        = azuread_application.grafana.application_id
  application_object_id = azuread_application.grafana.object_id

  resource_group_id = var.resource_group.id
  resource_group_roles = [
    "Monitoring Reader",
  ]
}

resource "azuread_application" "test_grafana" {
  display_name = "${var.resource_group.name}-test-grafana"
}
module "test_grafana_sp" {
  source = "../service_principal"

  name                  = "test-grafana"
  application_id        = azuread_application.test_grafana.application_id
  application_object_id = azuread_application.test_grafana.object_id

  resource_group_id = var.resource_group.id
  resource_group_roles = [
    "Monitoring Reader",
  ]
}
