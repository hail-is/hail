output "k8s_server_ip" {
  value = google_container_cluster.vdc.endpoint
}

output "oauth2_credentials" {
  value = jsondecode(file("~/.hail/auth_oauth2_client_secret.json"))
}
