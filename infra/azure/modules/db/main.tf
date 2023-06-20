resource "random_id" "db_name_suffix" {
  byte_length = 4
}

resource "random_password" "db_root_password" {
  length = 22
}

# This is required for mysql flexible server using a delegated_subnet
resource "azurerm_private_dns_zone" "db" {
  name                = "db.mysql.database.azure.com"
  resource_group_name = var.resource_group.name
}

resource "azurerm_private_dns_zone_virtual_network_link" "db" {
  name                  = "hail-db"
  private_dns_zone_name = azurerm_private_dns_zone.db.name
  virtual_network_id    = var.vnet_id
  resource_group_name   = var.resource_group.name
}

resource "azurerm_mysql_flexible_server" "db" {
  name                = "db-${random_id.db_name_suffix.hex}"
  resource_group_name = var.resource_group.name
  location            = var.resource_group.location

  administrator_login          = "mysqladmin"
  administrator_password = random_password.db_root_password.result

  version    = "8.0.21"
  sku_name   = "MO_Standard_E4ds_v4" # 4 vCPU, 8192 Mb per vCPU
  storage {
    auto_grow_enabled = true
  }

  # Which availability zone (out of 1,2,3) that the database should be hosted
  # in. This should ideally match the zone that batch is in but we don't have
  # availability zones enabled in AKS.
  # Sometimes zones are not available in particular regions
  # In this case either change to an appropriate zone or comment the below line out
  zone = 1

  delegated_subnet_id = var.subnet_id
  private_dns_zone_id = azurerm_private_dns_zone.db.id

  depends_on = [azurerm_private_dns_zone_virtual_network_link.db]
}

# Without this setting batch is not permitted to create
# MySQL functions since it is not SUPER
resource "azurerm_mysql_flexible_server_configuration" "trust_function_creators" {
  name                = "log_bin_trust_function_creators"
  resource_group_name = var.resource_group.name
  server_name         = azurerm_mysql_flexible_server.db.name
  value               = "ON"
}

resource "azurerm_mysql_flexible_server_configuration" "max_connections" {
  name                = "max_connections"
  resource_group_name = var.resource_group.name
  server_name         = azurerm_mysql_flexible_server.db.name
  value               = 1000
}

data "http" "db_ca_cert" {
  url = "https://dl.cacerts.digicert.com/DigiCertGlobalRootCA.crt.pem"
}

module "sql_config" {
  source = "../../../k8s/sql_config"

  server_ca_cert     = data.http.db_ca_cert.body
  host               = azurerm_mysql_flexible_server.db.fqdn
  user               = azurerm_mysql_flexible_server.db.administrator_login
  password           = azurerm_mysql_flexible_server.db.administrator_password
  instance           = azurerm_mysql_flexible_server.db.name
  connection_name    = azurerm_mysql_flexible_server.db.name
}
