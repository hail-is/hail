terraform {
  required_providers {
    google = {
      source = "hashicorp/google"
      version = "3.48.0"
    }
    kubernetes = {
      source = "hashicorp/kubernetes"
      version = "1.13.3"
    }
  }
}

variable "gsuite_organization" {}
variable "batch_gcp_regions" {}
variable "gcp_project" {}
variable "gcp_location" {}
variable "gcp_region" {}
variable "gcp_zone" {}
variable "domain" {}
variable "use_artifact_registry" { 
  type = bool
  description = "pull the ubuntu image from Artifact Registry. Otherwise, GCR"
}

locals {
  docker_prefix = (
    var.use_artifact_registry ?
    "${var.gcp_region}-docker.pkg.dev/${var.gcp_project}/hail" : 
    "gcr.io/${var.gcp_project}"
  )
  docker_root_image = "${local.docker_prefix}/ubuntu:18.04"
}

provider "google" {
  credentials = file("~/.hail/terraform_sa_key.json")

  project = var.gcp_project
  region = var.gcp_region
  zone = var.gcp_zone
}

provider "google-beta" {
  credentials = file("~/.hail/terraform_sa_key.json")

  project = var.gcp_project
  region = var.gcp_region
  zone = var.gcp_zone
}

data "google_client_config" "provider" {}

resource "google_project_service" "service_networking" {
  service = "servicenetworking.googleapis.com"
}

resource "google_compute_network" "default" {
  name = "default"
}

data "google_compute_subnetwork" "default_region" {
  name = "default"
  region = var.gcp_region
  depends_on = [google_compute_network.default]
}

resource "google_container_cluster" "vdc" {
  name = "vdc"
  location = var.gcp_zone
  network = google_compute_network.default.name

  # We can't create a cluster with no node pool defined, but we want to only use
  # separately managed node pools. So we create the smallest possible default
  # node pool and immediately delete it.
  remove_default_node_pool = true
  initial_node_count = 1

  master_auth {
    username = ""
    password = ""

    client_certificate_config {
      issue_client_certificate = false
    }
  }

  release_channel {
    channel = "STABLE"
  }
}

resource "google_container_node_pool" "vdc_preemptible_pool" {
  name = "preemptible-pool"
  location = var.gcp_zone
  cluster = google_container_cluster.vdc.name

  # Allocate at least one node, so that autoscaling can take place.
  initial_node_count = 1

  autoscaling {
    min_node_count = 0
    max_node_count = 200
  }

  node_config {
    preemptible = true
    machine_type = "n1-standard-2"

    labels = {
      "preemptible" = "true"
    }

    taint {
      key = "preemptible"
      value = "true"
      effect = "NO_SCHEDULE"
    }

    metadata = {
      disable-legacy-endpoints = "true"
    }

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
  }
}

resource "google_container_node_pool" "vdc_nonpreemptible_pool" {
  name = "nonpreemptible-pool"
  location = var.gcp_zone
  cluster = google_container_cluster.vdc.name

  # Allocate at least one node, so that autoscaling can take place.
  initial_node_count = 1

  autoscaling {
    min_node_count = 0
    max_node_count = 200
  }

  node_config {
    preemptible = false
    machine_type = "n1-standard-2"

    labels = {
      preemptible = "false"
    }

    metadata = {
      disable-legacy-endpoints = "true"
    }

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
  }
}

resource "random_id" "db_name_suffix" {
  byte_length = 4
}

# Without this, I get:
# Error: Error, failed to create instance because the network doesn't have at least 
# 1 private services connection. Please see 
# https://cloud.google.com/sql/docs/mysql/private-ip#network_requirements 
# for how to create this connection.
resource "google_compute_global_address" "google_managed_services_default" {
  name = "google-managed-services-default"
  purpose = "VPC_PEERING"
  address_type = "INTERNAL"
  prefix_length = 16
  network = google_compute_network.default.id
}

resource "google_service_networking_connection" "private_vpc_connection" {
  network = google_compute_network.default.id
  service = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.google_managed_services_default.name]
}

resource "google_sql_database_instance" "db" {
  name = "db-${random_id.db_name_suffix.hex}"
  database_version = "MYSQL_5_7"
  region = var.gcp_region

  depends_on = [google_service_networking_connection.private_vpc_connection]

  settings {
    # Second-generation instance tiers are based on the machine
    # type. See argument reference below.
    tier = "db-n1-standard-1"

    ip_configuration {
      ipv4_enabled = false
      private_network = google_compute_network.default.id
      require_ssl = true
    }
  }
}

resource "google_compute_address" "gateway" {
  name = "gateway"
  region = var.gcp_region
}

resource "google_compute_address" "internal_gateway" {
  name = "internal-gateway"
  subnetwork = data.google_compute_subnetwork.default_region.id
  address_type = "INTERNAL"
  region = var.gcp_region
}

provider "kubernetes" {
  load_config_file = false

  host = "https://${google_container_cluster.vdc.endpoint}"
  token = data.google_client_config.provider.access_token
  cluster_ca_certificate = base64decode(
    google_container_cluster.vdc.master_auth[0].cluster_ca_certificate,
  )
}

resource "kubernetes_secret" "global_config" {
  metadata {
    name = "global-config"
  }

  data = {
    batch_gcp_regions = var.batch_gcp_regions
    batch_logs_bucket = google_storage_bucket.batch_logs.name
    default_namespace = "default"
    docker_root_image = local.docker_root_image
    domain = var.domain
    gcp_project = var.gcp_project
    gcp_region = var.gcp_region
    gcp_zone = var.gcp_zone
    docker_prefix = local.docker_prefix
    gsuite_organization = var.gsuite_organization
    internal_ip = google_compute_address.internal_gateway.address
    ip = google_compute_address.gateway.address
    kubernetes_server_url = "https://${google_container_cluster.vdc.endpoint}"
  }
}

resource "google_sql_ssl_cert" "root_client_cert" {
  common_name = "root-client-cert"
  instance = google_sql_database_instance.db.name
}

resource "random_password" "db_root_password" {
  length = 22
}

resource "google_sql_user" "db_root" {
  name = "root"
  instance = google_sql_database_instance.db.name
  password = random_password.db_root_password.result
}

resource "kubernetes_secret" "database_server_config" {
  metadata {
    name = "database-server-config"
  }

  data = {
    "server-ca.pem" = google_sql_database_instance.db.server_ca_cert.0.cert
    "client-cert.pem" = google_sql_ssl_cert.root_client_cert.cert
    "client-key.pem" = google_sql_ssl_cert.root_client_cert.private_key
    "sql-config.cnf" = <<END
[client]
host=${google_sql_database_instance.db.ip_address[0].ip_address}
user=root
password=${random_password.db_root_password.result}
ssl-ca=/sql-config/server-ca.pem
ssl-cert=/sql-config/client-cert.pem
ssl-key=/sql-config/client-key.pem
ssl-mode=VERIFY_CA
END
    "sql-config.json" = <<END
{
    "docker_root_image": "${local.docker_root_image}",
    "host": "${google_sql_database_instance.db.ip_address[0].ip_address}",
    "port": 3306,
    "user": "root",
    "password": "${random_password.db_root_password.result}",
    "instance": "${google_sql_database_instance.db.name}",
    "connection_name": "${google_sql_database_instance.db.connection_name}",
    "ssl-ca": "/sql-config/server-ca.pem",
    "ssl-cert": "/sql-config/client-cert.pem",
    "ssl-key": "/sql-config/client-key.pem",
    "ssl-mode": "VERIFY_CA"
}
END
  }
}

resource "google_container_registry" "registry" {
}

resource "google_artifact_registry_repository" "repository" {
  provider = google-beta
  format = "DOCKER"
  repository_id = "hail"
  location = var.gcp_location
}

resource "google_service_account" "gcr_pull" {
  account_id = "gcr-pull"
  display_name = "pull from gcr.io"
}

resource "google_service_account_key" "gcr_pull_key" {
  service_account_id = google_service_account.gcr_pull.name
}

resource "google_service_account" "gcr_push" {
  account_id = "gcr-push"
  display_name = "push to gcr.io"
}

resource "google_service_account_key" "gcr_push_key" {
  service_account_id = google_service_account.gcr_push.name
}

resource "google_storage_bucket_iam_member" "gcr_pull_viewer" {
  bucket = google_container_registry.registry.id
  role = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.gcr_pull.email}"
}

resource "google_artifact_registry_repository_iam_member" "artifact_registry_pull_viewer" {
  provider = google-beta
  repository = google_artifact_registry_repository.repository.name
  location = var.gcp_location
  role = "roles/artifactregistry.reader"
  member = "serviceAccount:${google_service_account.gcr_pull.email}"
}

resource "google_artifact_registry_repository_iam_member" "artifact_registry_ci_viewer" {
  provider = google-beta
  repository = google_artifact_registry_repository.repository.name
  location = var.gcp_location
  role = "roles/artifactregistry.reader"
  member = "serviceAccount:${google_service_account.ci.email}"
}

resource "google_storage_bucket_iam_member" "gcr_push_admin" {
  bucket = google_container_registry.registry.id
  role = "roles/storage.admin"
  member = "serviceAccount:${google_service_account.gcr_push.email}"
}

resource "google_artifact_registry_repository_iam_member" "artifact_registry_push_admin" {
  provider = google-beta
  repository = google_artifact_registry_repository.repository.name
  location = var.gcp_location
  role = "roles/artifactregistry.admin"
  member = "serviceAccount:${google_service_account.gcr_push.email}"
}

resource "kubernetes_secret" "gcr_pull_key" {
  metadata {
    name = "gcr-pull-key"
  }

  data = {
    "gcr-pull.json" = base64decode(google_service_account_key.gcr_pull_key.private_key)
  }
}

resource "kubernetes_secret" "gcr_push_key" {
  metadata {
    name = "gcr-push-service-account-key"
  }

  data = {
    "gcr-push-service-account-key.json" = base64decode(google_service_account_key.gcr_push_key.private_key)
  }
}

resource "kubernetes_namespace" "ukbb_rg" {
  metadata {
    name = "ukbb-rg"
  }
}

resource "kubernetes_service" "ukbb_rb_browser" {
  metadata {
    name = "ukbb-rg-browser"
    namespace = kubernetes_namespace.ukbb_rg.metadata[0].name
    labels = {
      app = "ukbb-rg-browser"
    }
  }
  spec {
    port {
      port = 80
      protocol = "TCP"
      target_port = 80
    }
    selector = {
      app = "ukbb-rg-browser"
    }
  }
}

resource "kubernetes_service" "ukbb_rb_static" {
  metadata {
    name = "ukbb-rg-static"
    namespace = kubernetes_namespace.ukbb_rg.metadata[0].name
    labels = {
      app = "ukbb-rg-static"
    }
  }
  spec {
    port {
      port = 80
      protocol = "TCP"
      target_port = 80
    }
    selector = {
      app = "ukbb-rg-static"
    }
  }
}

resource "random_id" "atgu_name_suffix" {
  byte_length = 2
}

resource "google_service_account" "atgu" {
  account_id = "atgu-${random_id.atgu_name_suffix.hex}"
}

resource "google_service_account_key" "atgu_key" {
  service_account_id = google_service_account.atgu.name
}

resource "kubernetes_secret" "atgu_gsa_key" {
  metadata {
    name = "atgu-gsa-key"
  }

  data = {
    "key.json" = base64decode(google_service_account_key.atgu_key.private_key)
  }
}

resource "random_id" "auth_name_suffix" {
  byte_length = 2
}

resource "google_service_account" "auth" {
  account_id = "auth-${random_id.auth_name_suffix.hex}"
}

resource "google_service_account_key" "auth_key" {
  service_account_id = google_service_account.auth.name
}

resource "google_project_iam_member" "auth_service_account_admin" {
  role = "roles/iam.serviceAccountAdmin"
  member = "serviceAccount:${google_service_account.auth.email}"
}

resource "google_project_iam_member" "auth_service_account_key_admin" {
  role = "roles/iam.serviceAccountKeyAdmin"
  member = "serviceAccount:${google_service_account.auth.email}"
}

resource "google_project_iam_member" "auth_storage_admin" {
  role = "roles/storage.admin"
  member = "serviceAccount:${google_service_account.auth.email}"
}

resource "kubernetes_secret" "auth_gsa_key" {
  metadata {
    name = "auth-gsa-key"
  }

  data = {
    "key.json" = base64decode(google_service_account_key.auth_key.private_key)
  }
}

resource "random_id" "batch_name_suffix" {
  byte_length = 2
}

resource "google_service_account" "batch" {
  account_id = "batch-${random_id.batch_name_suffix.hex}"
}

resource "google_service_account_key" "batch_key" {
  service_account_id = google_service_account.batch.name
}

resource "kubernetes_secret" "batch_gsa_key" {
  metadata {
    name = "batch-gsa-key"
  }

  data = {
    "key.json" = base64decode(google_service_account_key.batch_key.private_key)
  }
}

resource "google_project_iam_member" "batch_compute_instance_admin" {
  role = "roles/compute.instanceAdmin.v1"
  member = "serviceAccount:${google_service_account.batch.email}"
}

resource "google_project_iam_member" "batch_service_account_user" {
  role = "roles/iam.serviceAccountUser"
  member = "serviceAccount:${google_service_account.batch.email}"
}

resource "google_project_iam_member" "batch_logging_viewer" {
  role = "roles/logging.viewer"
  member = "serviceAccount:${google_service_account.batch.email}"
}

resource "google_project_iam_member" "batch_storage_admin" {
  role = "roles/storage.admin"
  member = "serviceAccount:${google_service_account.batch.email}"
}

resource "google_service_account" "benchmark" {
  account_id = "benchmark"
}

resource "google_service_account_key" "benchmark_key" {
  service_account_id = google_service_account.benchmark.name
}

resource "kubernetes_secret" "benchmark_gsa_key" {
  metadata {
    name = "benchmark-gsa-key"
  }

  data = {
    "key.json" = base64decode(google_service_account_key.benchmark_key.private_key)
  }
}

resource "random_id" "ci_name_suffix" {
  byte_length = 2
}

resource "google_service_account" "ci" {
  account_id = "ci-${random_id.ci_name_suffix.hex}"
}

resource "google_artifact_registry_repository_iam_member" "artifact_registry_viewer" {
  provider = google-beta
  repository = google_artifact_registry_repository.repository.name
  location = var.gcp_location
  role = "roles/artifactregistry.reader"
  member = "serviceAccount:${google_service_account.ci.email}"
}

resource "google_service_account_key" "ci_key" {
  service_account_id = google_service_account.ci.name
}

resource "kubernetes_secret" "ci_gsa_key" {
  metadata {
    name = "ci-gsa-key"
  }

  data = {
    "key.json" = base64decode(google_service_account_key.ci_key.private_key)
  }
}

resource "google_service_account" "monitoring" {
  account_id = "monitoring"
}

resource "google_service_account_key" "monitoring_key" {
  service_account_id = google_service_account.monitoring.name
}

resource "kubernetes_secret" "monitoring_gsa_key" {
  metadata {
    name = "monitoring-gsa-key"
  }

  data = {
    "key.json" = base64decode(google_service_account_key.monitoring_key.private_key)
  }
}

resource "random_id" "test_name_suffix" {
  byte_length = 2
}

resource "google_service_account" "test" {
  account_id = "test-${random_id.test_name_suffix.hex}"
}

resource "google_service_account_key" "test_key" {
  service_account_id = google_service_account.test.name
}

resource "kubernetes_secret" "test_gsa_key" {
  metadata {
    name = "test-gsa-key"
  }

  data = {
    "key.json" = base64decode(google_service_account_key.test_key.private_key)
  }
}

resource "google_project_iam_member" "test_compute_instance_admin" {
  role = "roles/compute.instanceAdmin.v1"
  member = "serviceAccount:${google_service_account.test.email}"
}

resource "google_project_iam_member" "test_service_account_user" {
  role = "roles/iam.serviceAccountUser"
  member = "serviceAccount:${google_service_account.test.email}"
}

resource "google_project_iam_member" "test_logging_viewer" {
  role = "roles/logging.viewer"
  member = "serviceAccount:${google_service_account.test.email}"
}

resource "google_project_iam_member" "test_service_usage_consumer" {
  role = "roles/serviceusage.serviceUsageConsumer"
  member = "serviceAccount:${google_service_account.test.email}"
}

resource "google_service_account" "test_dev" {
  account_id = "test-dev"
}

resource "google_service_account_key" "test_dev_key" {
  service_account_id = google_service_account.test_dev.name
}

resource "kubernetes_secret" "test_dev_gsa_key" {
  metadata {
    name = "test-dev-gsa-key"
  }

  data = {
    "key.json" = base64decode(google_service_account_key.test_dev_key.private_key)
  }
}

resource "google_service_account" "batch_agent" {
  account_id = "batch2-agent"
}

resource "google_project_iam_member" "batch_agent_compute_instance_admin" {
  role = "roles/compute.instanceAdmin.v1"
  member = "serviceAccount:${google_service_account.batch_agent.email}"
}

resource "google_project_iam_member" "batch_agent_service_account_user" {
  role = "roles/iam.serviceAccountUser"
  member = "serviceAccount:${google_service_account.batch_agent.email}"
}

resource "google_project_iam_member" "batch_agent_log_writer" {
  role = "roles/logging.logWriter"
  member = "serviceAccount:${google_service_account.batch_agent.email}"
}

resource "google_project_iam_member" "batch_agent_object_creator" {
  role = "roles/storage.objectCreator"
  member = "serviceAccount:${google_service_account.batch_agent.email}"
}

resource "google_project_iam_member" "batch_agent_object_viewer" {
  role = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.batch_agent.email}"
}

resource "google_compute_firewall" "default_allow_internal" {
  name    = "default-allow-internal"
  network = google_compute_network.default.name

  priority = 65534

  source_ranges = ["10.128.0.0/9"]

  allow {
    protocol = "tcp"
    ports    = ["0-65535"]
  }

  allow {
    protocol = "udp"
    ports    = ["0-65535"]
  }

  allow {
    protocol = "icmp"
  }
}

resource "google_compute_firewall" "allow_ssh" {
  name    = "allow-ssh"
  network = google_compute_network.default.name

  priority = 65534

  source_ranges = ["0.0.0.0/0"]

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }
}

resource "google_compute_firewall" "vdc_to_batch_worker" {
  name    = "vdc-to-batch-worker"
  network = google_compute_network.default.name

  source_ranges = [google_container_cluster.vdc.cluster_ipv4_cidr]

  target_tags = ["batch2-agent"]

  allow {
    protocol = "icmp"
  }

  allow {
    protocol = "tcp"
    ports    = ["1-65535"]
  }

  allow {
    protocol = "udp"
    ports    = ["1-65535"]
  }
}

resource "random_id" "batch_logs_bucket_name_suffix" {
  byte_length = 2
}

resource "google_storage_bucket" "batch_logs" {
  name = "batch-logs-${random_id.batch_logs_bucket_name_suffix.hex}"
  location = var.gcp_location
  force_destroy = true
  storage_class = "STANDARD"
}

resource "google_dns_managed_zone" "dns_zone" {
  name = "dns-zone"
  dns_name = "hail."
  visibility = "private"

  private_visibility_config {
    networks {
      network_url = google_compute_network.default.id
    }
  }
}

resource "google_dns_record_set" "internal_gateway" {
  name = "*.${google_dns_managed_zone.dns_zone.dns_name}"
  managed_zone = google_dns_managed_zone.dns_zone.name
  type = "A"
  ttl = 300

  rrdatas = [google_compute_address.internal_gateway.address]
}
