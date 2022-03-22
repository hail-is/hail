output kube_config {
  value = azurerm_kubernetes_cluster.vdc.kube_config[0]
}

output kubernetes_server {
  value = azurerm_kubernetes_cluster.vdc.fqdn
}

output gateway_ip {
  value = azurerm_public_ip.gateway_ip.ip_address
}

output internal_gateway_ip {
  # An IP toward the top of the batch-worker-subnet IP address range
  # that even in an existing cluster should be unused. In Azure we can't
  # explicitly reserve an internal IP, but once this is used the first time
  # for internal-gateway it won't be revoked and reused.
  value = "10.128.255.254"
}

output vnet_id {
  value = azurerm_virtual_network.default.id
}

output db_subnet_id {
  value = azurerm_subnet.db_subnet.id
}
