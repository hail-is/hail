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
    }
    azure = {
      azure_resource_group = data.azurerm_resource_group.rg.name
      azure_subscription_id = var.subscription_id
      azure_location = data.azurerm_resource_group.rg.location
    }
  }
}

output "sql_config" {
  value = {
    server_ca_cert = data.http.db_ca_cert.body
    client_cert = ""
    client_private_key = ""
    host = azurerm_private_endpoint.db_endpoint.private_service_connection[0].private_ip_address
    user = "${azurerm_mysql_server.db.administrator_login}@${azurerm_mysql_server.db.name}"
    password = azurerm_mysql_server.db.administrator_login_password
  }
  sensitive = true
}
