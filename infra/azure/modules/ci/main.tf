resource "azurerm_storage_account" "ci" {
  name                     = "${var.resource_group.name}ci${var.storage_account_suffix}"
  resource_group_name      = var.resource_group.name
  location                 = var.resource_group.location
  account_tier             = "Standard"
  account_replication_type = "LRS"

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

module "k8s_resources" {
  source = "../../../k8s/ci"

  storage_uri                             = "hail-az://${azurerm_storage_account.ci.name}/${azurerm_storage_container.ci_artifacts.name}"
  deploy_steps                            = var.deploy_steps
  watched_branches                        = var.watched_branches
  test_oauth2_callback_urls               = var.test_oauth2_callback_urls
  github_context                          = var.github_context
  ci_and_deploy_github_oauth_token        = var.ci_and_deploy_github_oauth_token
  ci_test_repo_creator_github_oauth_token = var.ci_test_repo_creator_github_oauth_token
}
