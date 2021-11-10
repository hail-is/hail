output "storage_uri" {
  value = "hail-az://${azurerm_storage_account.ci.name}/${azurerm_storage_container.ci_artifacts.name}"
}
