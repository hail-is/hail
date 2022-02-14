variable server_ca_cert {}
variable client_cert {
  type    = string
  default = ""
}
variable client_private_key {
  type    = string
  default = ""
}
variable host {}
variable user {}
variable password {}
variable instance {}
variable connection_name {}

locals {
  config_cnf = <<END
[client]
host=${var.host}
user=${var.user}
password=${var.password}
ssl-ca=/sql-config/server-ca.pem
ssl-mode=VERIFY_CA
END

  config_json = {
    ssl-ca          = "/sql-config/server-ca.pem"
    ssl-mode        = "VERIFY_CA"
    host            = var.host
    instance        = var.instance
    connection_name = var.connection_name
    port            = 3306
    user            = var.user
    password        = var.password
  }
}

resource "kubernetes_secret" "mtls_database_server_config" {
  count = var.client_cert != "" ? 1 : 0

  metadata {
    name = "database-server-config"
  }

  data = {
    "server-ca.pem" = var.server_ca_cert
    "client-cert.pem" = var.client_cert
    "client-key.pem" = var.client_private_key
    "sql-config.cnf" = <<END
${local.config_cnf}
ssl-cert=${var.client_cert}
ssl-key=${var.client_private_key}
END
    "sql-config.json" = jsonencode(merge({
      "ssl-cert" = "/sql-config/client-cert.pem"
      "ssl-key"  = "/sql-config/client-key.pem"
    }, local.config_json))
  }

}

resource "kubernetes_secret" "database_server_config" {
  count = var.client_cert == "" ? 1 : 0
  metadata {
    name = "database-server-config"
  }

  data = {
    "server-ca.pem"   = var.server_ca_cert
    "sql-config.cnf"  = local.config_cnf
    "sql-config.json" = jsonencode(local.config_json)
  }
}
