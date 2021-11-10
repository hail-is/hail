resource "azurerm_storage_account" "ci" {
  name                     = "${var.resource_group.name}ci"
  resource_group_name      = var.resource_group.name
  location                 = var.resource_group.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
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
      blob_types   = ["blockBlob", "appendBlob"]
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

resource "azurerm_role_assignment" "ci_acr_role" {
  for_each = toset(["AcrPush", "AcrDelete"])

  scope                = var.container_registry_id
  role_definition_name = each.key
  principal_id         = var.ci_principal_id
}
