output "kube_config" {
  value     = google_container_cluster.vdc
  sensitive = true
}

output "global_config" {
  value = {
    global = {
      cloud = "gcp"
      domain = var.domain
      docker_prefix = local.docker_prefix
      internal_ip = google_compute_address.internal_gateway.address
      ip = google_compute_address.gateway.address
      kubernetes_server = google_container_cluster.vdc.endpoint
      ci_watched_branches = var.ci_watched_branches
      ci_deploy_steps = var.ci_deploy_steps
    }
    azure = {}
    gcp = {
      batch_gcp_regions = var.batch_gcp_regions
      batch_logs_bucket = module.batch_logs.name
      hail_query_gcs_path = "gs://${module.hail_query.name}"
      hail_ci_gcs_bucket = module.hail_ci_bucket.name
      hail_test_gcs_bucket = module.hail_test_gcs_bucket.name
      hail_test_requester_pays_gcs_bucket = module.hail_test_requester_pays_gcs_bucket.name
      gcp_project = var.gcp_project
      gcp_region = var.gcp_region
      gcp_zone = var.gcp_zone
      gsuite_organization = var.gsuite_organization
    }
  }
}

output "sql_config" {
  value = {
    server_ca_cert = google_sql_database_instance.db.server_ca_cert.0.cert
    client_cert = google_sql_ssl_cert.root_client_cert.cert
    client_private_key = google_sql_ssl_cert.root_client_cert.private_key
    host = google_sql_database_instance.db.ip_address[0].ip_address
    user = "root"
    password = random_password.db_root_password.result
  }
  sensitive = true
}
