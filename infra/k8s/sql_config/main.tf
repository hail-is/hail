variable server_ca_cert {}
variable client_cert {}
variable client_private_key {}
variable host {}
variable user {}
variable password {}
variable instance {}
variable connection_name {}

resource "kubernetes_secret" "database_server_config" {
  metadata {
    name = "database-server-config"
  }

  data = {
    "server-ca.pem" = var.server_ca_cert
    "client-cert.pem" = var.client_cert
    "client-key.pem" = var.client_private_key
    "sql-config.cnf" = <<END
[client]
host=${var.host}
user=${var.user}
password=${var.password}
ssl-ca=/sql-config/server-ca.pem
ssl-mode=VERIFY_CA
ssl-cert=/sql-config/client-cert.pem
ssl-key=/sql-config/client-key.pem
END
    "sql-config.json" = <<END
{
    "ssl-cert": "/sql-config/client-cert.pem",
    "ssl-key": "/sql-config/client-key.pem",
    "ssl-ca": "/sql-config/server-ca.pem",
    "ssl-mode": "VERIFY_CA",
    "host": "${var.host}",
    "instance": "${var.instance}",
    "connection_name": "${var.connection_name}",
    "port": 3306,
    "user": "${var.user}",
    "password": "${var.password}"
}
END
  }
}
