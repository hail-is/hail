resource "kubernetes_secret" "database_server_config" {
  metadata {
    name = "database-server-config"
  }

  data = {
    "server-ca.pem" = var.sql_config.server_ca_cert
    "client-cert.pem" = var.sql_config.client_cert
    "client-key.pem" = var.sql_config.client_private_key
    "sql-config.cnf" = <<END
[client]
host=${var.sql_config.host}
user=${var.sql_config.user}
password=${var.sql_config.password}
ssl-ca=/sql-config/server-ca.pem
ssl-mode=VERIFY_CA
%{ if var.sql_config.client_cert != "" }
ssl-cert=/sql-config/client-cert.pem
ssl-key=/sql-config/client-key.pem
%{ endif }
END
    "sql-config.json" = <<END
{
%{ if var.sql_config.client_cert != "" }
    "ssl-cert": "/sql-config/client-cert.pem",
    "ssl-key": "/sql-config/client-key.pem",
%{ endif }
    "ssl-ca": "/sql-config/server-ca.pem",
    "ssl-mode": "VERIFY_CA",
    "host": "${var.sql_config.host}",
    "port": 3306,
    "user": "${var.sql_config.user}",
    "password": "${var.sql_config.password}",
    "docker_root_image": "${local.docker_root_image}"
}
END
  }
}
