resource "random_id" "db_name_suffix" {
  byte_length = 4
}

resource "random_password" "db_root_password" {
  length = 22
}

resource "azurerm_mysql_server" "db" {
  name                = "db-${random_id.db_name_suffix.hex}"
  resource_group_name = var.resource_group.name
  location            = var.resource_group.location

  administrator_login          = "mysqladmin"
  administrator_login_password = random_password.db_root_password.result

  version    = "5.7"
  sku_name   = "MO_Gen5_2" # 2 vCPU, 10GiB per vCPU
  storage_mb = 5120

  ssl_enforcement_enabled       = true
  public_network_access_enabled = false
}

# Without this setting batch is not permitted to create
# MySQL functions since it is not SUPER
resource "azurerm_mysql_configuration" "example" {
  name                = "log_bin_trust_function_creators"
  resource_group_name = var.resource_group.name
  server_name         = azurerm_mysql_server.db.name
  value               = "ON"
}

resource "azurerm_mysql_configuration" "max_connections" {
  name                = "max_connections"
  resource_group_name = var.resource_group.name
  server_name         = azurerm_mysql_server.db.name
  value               = 1000
}

data "http" "db_ca_cert" {
  url = "https://www.digicert.com/CACerts/BaltimoreCyberTrustRoot.crt.pem"
}

resource "azurerm_private_endpoint" "db_endpoint" {
  name                = "${azurerm_mysql_server.db.name}-endpoint"
  resource_group_name = var.resource_group.name
  location            = var.resource_group.location
  subnet_id           = var.subnet_id

  private_service_connection {
    name                           = "${azurerm_mysql_server.db.name}-endpoint"
    private_connection_resource_id = azurerm_mysql_server.db.id
    subresource_names              = [ "mysqlServer" ]
    is_manual_connection           = false
  }
}

resource "tls_private_key" "db_client_key" {
  algorithm = "RSA"
}

resource "tls_self_signed_cert" "db_client_cert" {
  key_algorithm   = tls_private_key.db_client_key.algorithm
  private_key_pem = tls_private_key.db_client_key.private_key_pem

  subject {
    common_name  = "hail-client"
  }

  validity_period_hours = 24 * 365

  allowed_uses = [
    "client_auth"
  ]
}

module "sql_config" {
  source = "../../../k8s/sql_config"

  server_ca_cert     = data.http.db_ca_cert.body
  client_cert        = tls_self_signed_cert.db_client_cert.cert_pem
  client_private_key = tls_private_key.db_client_key.private_key_pem
  host               = azurerm_private_endpoint.db_endpoint.private_service_connection[0].private_ip_address
  user               = "${azurerm_mysql_server.db.administrator_login}@${azurerm_mysql_server.db.name}"
  password           = azurerm_mysql_server.db.administrator_login_password
  instance           = azurerm_mysql_server.db.name
  connection_name    = azurerm_mysql_server.db.name
}
