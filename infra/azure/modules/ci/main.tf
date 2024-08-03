data "azurerm_client_config" "primary" {}

resource "azurerm_storage_account" "ci" {
  name                     = "${var.resource_group.name}ci${var.storage_account_suffix}"
  resource_group_name      = var.resource_group.name
  location                 = var.resource_group.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  min_tls_version          = "TLS1_0"

  allow_nested_items_to_be_public   = false

  blob_properties {
    last_access_time_enabled = true
  }
}

resource "azurerm_storage_container" "ci_artifacts" {
  name                  = "artifacts"
  storage_account_name  = azurerm_storage_account.ci.name
  container_access_type = "private"
}

resource "azurerm_storage_management_policy" "ci" {
  storage_account_id = azurerm_storage_account.ci.id

  rule {
    name    = "ci-artifacts-retention-7-day"
    enabled = true
    filters {
      prefix_match = [azurerm_storage_container.ci_artifacts.name]
      blob_types   = ["blockBlob"]
    }
    actions {
      base_blob {
        delete_after_days_since_modification_greater_than = 7
      }
    }
  }
}

resource "azurerm_role_assignment" "ci_ci_account_contributor" {
  scope                = azurerm_storage_account.ci.id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = var.ci_principal_id
}

resource "azurerm_role_assignment" "ci_test_container_contributor" {
  scope                = var.test_storage_container_resource_id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = var.ci_principal_id
}

resource "azurerm_key_vault_access_policy" "ci_manage" {
  key_vault_id = var.worker_key_vault_id
  tenant_id    = data.azurerm_client_config.primary.tenant_id
  object_id    = var.ci_principal_id

  secret_permissions = [
    "Get", "List", "Delete"
  ]
}

module "k8s_resources" {
  source = "../../../k8s/ci"

  storage_uri                             = "https://${azurerm_storage_account.ci.name}.blob.core.windows.net/${azurerm_storage_container.ci_artifacts.name}"
  deploy_steps                            = var.deploy_steps
  watched_branches                        = var.watched_branches
  github_context                          = var.github_context
  ci_and_deploy_github_oauth_token        = var.ci_and_deploy_github_oauth_token
  ci_test_repo_creator_github_oauth_token = var.ci_test_repo_creator_github_oauth_token
}
