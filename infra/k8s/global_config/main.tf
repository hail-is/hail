variable cloud {}
variable domain {}
variable organization_domain {}
variable internal_gateway_ip {}
variable gateway_ip {}
variable kubernetes_server {}
variable docker_prefix {}
variable batch_logs_storage_uri {}
variable test_storage_uri {}
variable query_storage_uri {}
variable extra_fields {
  type = map
}

resource "kubernetes_secret" "global_config" {
  metadata {
    name = "global-config"
  }

  data = merge({
    cloud                  = var.cloud
    default_namespace      = "default"
    docker_prefix          = var.docker_prefix
    docker_root_image      = "${var.docker_prefix}/ubuntu:22.04"
    domain                 = var.domain
    organization_domain    = var.organization_domain
    internal_ip            = var.internal_gateway_ip
    ip                     = var.gateway_ip
    kubernetes_server_url  = "https://${var.kubernetes_server}"
    batch_logs_storage_uri = var.batch_logs_storage_uri
    test_storage_uri       = var.test_storage_uri
    query_storage_uri      = var.query_storage_uri
  }, var.extra_fields)
}
