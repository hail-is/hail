resource "kubernetes_secret" "global_config" {
  metadata {
    name = "global-config"
  }

  data = merge({
    cloud                 = var.global_config.cloud
    default_namespace     = "default"
    docker_prefix         = var.global_config.docker_prefix
    docker_root_image     = local.docker_root_image
    domain                = var.global_config.domain
    internal_ip           = var.global_config.internal_ip
    ip                    = var.global_config.ip
    kubernetes_server_url = "https://${var.global_config.kubernetes_server}"
  }, var.global_config.azure)
}
