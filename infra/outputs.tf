output "k8s_server_ip" {
  value = google_container_cluster.vdc.endpoint
}
