output batch_logs_storage_uri {
  value = "https://${azurerm_storage_account.batch.name}.blob.core.windows.net/${azurerm_storage_container.batch_logs.name}"
}

output query_storage_uri {
  value = "https://${azurerm_storage_account.batch.name}.blob.core.windows.net/${azurerm_storage_container.query.name}"
}

output test_storage_container {
  value = azurerm_storage_container.test
}

output test_storage_uri {
  value = "https://${azurerm_storage_account.test.name}.blob.core.windows.net/${azurerm_storage_container.test.name}"
}

output ci_principal_id {
  value = module.ci_sp.principal_id
}
