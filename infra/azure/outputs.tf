output "kube_config" {
  value     = azurerm_kubernetes_cluster.vdc.kube_config_raw
  sensitive = true
}

output "global_config" {
  value = {
    global = {
      cloud = "azure"
      domain = var.domain
      docker_prefix = azurerm_container_registry.acr.login_server
      internal_ip = local.internal_ip
      ip = azurerm_public_ip.gateway_ip.ip_address
      kubernetes_server = azurerm_kubernetes_cluster.vdc.fqdn
      batch_logs_storage_uri = "hail-az://${azurerm_storage_account.batch.name}/${azurerm_storage_container.batch_logs.name}"
      test_storage_uri = "hail-az://${azurerm_storage_account.test.name}/${azurerm_storage_container.test.name}"
    }
    azure = {
      # FIXME Just for testing
      gsuite_organization = "broadinstitute.org"
      azure_resource_group = data.azurerm_resource_group.rg.name
      azure_subscription_id = var.subscription_id
      azure_location = data.azurerm_resource_group.rg.location
    }
  }
}

output "sql_config" {
  value = {
    server_ca_cert = data.http.db_ca_cert.body
    client_cert = tls_self_signed_cert.db_client_cert.cert_pem
    client_private_key = tls_private_key.db_client_key.private_key_pem
    host = azurerm_private_endpoint.db_k8s_endpoint.private_service_connection[0].private_ip_address
    user = "${azurerm_mysql_server.db.administrator_login}@${azurerm_mysql_server.db.name}"
    password = azurerm_mysql_server.db.administrator_login_password
    instance = azurerm_mysql_server.db.name
    connection_name = azurerm_mysql_server.db.name
  }
  sensitive = true
}

output "registry_push_credentials" {
  value = {
    appId = azurerm_container_registry.acr.admin_username
    password = azurerm_container_registry.acr.admin_password
  }
  sensitive = true
}

output "service_credentials" {
  value = {
    auth      = module.auth_sp.credentials
    batch     = module.batch_sp.credentials
    benchmark = module.benchmark_sp.credentials
    ci        = module.ci_sp.credentials
    test      = module.test_sp.credentials
    test-dev  = module.test_dev_sp.credentials
    query     = module.query_sp.credentials
    grafana   = module.grafana_sp.credentials
  }
  sensitive = true
}
