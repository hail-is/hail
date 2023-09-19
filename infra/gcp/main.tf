terraform {
  required_providers {
    google = {
      source = "hashicorp/google"
      version = "4.32.0"
    }
    kubernetes = {
      source = "hashicorp/kubernetes"
      version = "2.8.0"
    }
    sops = {
      source = "carlpett/sops"
      version = "0.6.3"
    }
  }
  backend "gcs" {
    bucket = "cpg-hail-terraform"
  }
}

variable "k8s_preemptible_node_pool_name" {
  type    = string
  default = "preemptible-pool"
}
variable "k8s_nonpreemptible_node_pool_name" {
  type    = string
  default = "nonpreemptible-pool"
}
variable "batch_gcp_regions" {}
variable "gcp_project" {}
variable "batch_logs_bucket_location" {}
variable "batch_logs_bucket_storage_class" {}
variable "hail_query_bucket_location" {}
variable "hail_query_bucket_storage_class" {}
variable "hail_test_gcs_bucket_location" {}
variable "hail_test_gcs_bucket_storage_class" {}
variable "gcp_region" {}
variable "gcp_zone" {}
variable "gcp_location" {}
variable "domain" {}
variable "organization_domain" {}
variable "github_organization" {}
variable "use_artifact_registry" {
  type = bool
  description = "pull the ubuntu image from Artifact Registry. Otherwise, GCR"
}

variable deploy_ukbb {
  type = bool
  description = "Run the UKBB Genetic Correlation browser"
  default = false
}

locals {
  docker_prefix = (
    var.use_artifact_registry ?
    "${var.gcp_region}-docker.pkg.dev/${var.gcp_project}/hail" :
    "gcr.io/${var.gcp_project}"
  )
  docker_root_image = "${local.docker_prefix}/ubuntu:22.04"
}

data "sops_file" "terraform_sa_key_sops" {
  source_file = "${var.github_organization}/terraform_sa_key.enc.json"
}

provider "google" {
  credentials = data.sops_file.terraform_sa_key_sops.raw

  project = var.gcp_project
  region = var.gcp_region
  zone = var.gcp_zone
}

provider "google-beta" {
  credentials = data.sops_file.terraform_sa_key_sops.raw

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
  provider = google-beta
  name = "vdc"
  location = var.gcp_zone
  network = google_compute_network.default.name

  # We can't create a cluster with no node pool defined, but we want to only use
  # separately managed node pools. So we create the smallest possible default
  # node pool and immediately delete it.
  remove_default_node_pool = true
  initial_node_count = 1

  master_auth {
    client_certificate_config {
      issue_client_certificate = false
    }
  }

  release_channel {
    channel = "STABLE"
  }

  cluster_autoscaling {
    # Don't use node auto-provisioning since we manage node pools ourselves
    enabled = false
    autoscaling_profile = "OPTIMIZE_UTILIZATION"
  }
}

resource "google_container_node_pool" "vdc_preemptible_pool" {
  name     = var.k8s_preemptible_node_pool_name
  location = var.gcp_zone
  cluster  = google_container_cluster.vdc.name

  # Allocate at least one node, so that autoscaling can take place.
  initial_node_count = 1

  autoscaling {
    min_node_count = 0
    max_node_count = 200
  }

  node_config {
    spot = true
    machine_type = "n1-standard-4"

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
  name     = var.k8s_nonpreemptible_node_pool_name
  location = var.gcp_zone
  cluster  = google_container_cluster.vdc.name

  # Allocate at least one node, so that autoscaling can take place.
  initial_node_count = 1

  autoscaling {
    min_node_count = 0
    max_node_count = 200
  }

  node_config {
    preemptible = false
    machine_type = "n1-standard-4"

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

resource "google_compute_network_peering_routes_config" "private_vpc_peering_config" {
  peering = google_service_networking_connection.private_vpc_connection.peering
  network = google_compute_network.default.name
  import_custom_routes = true
  export_custom_routes = true
}

resource "google_sql_database_instance" "db" {
  name = "db-${random_id.db_name_suffix.hex}"
  database_version = "MYSQL_8_0"
  region = var.gcp_region

  depends_on = [google_service_networking_connection.private_vpc_connection]

  settings {
    # 4 vCPU and 15360 MB
    # https://cloud.google.com/sql/docs/mysql/instance-settings
    tier = "db-custom-4-15360"

    ip_configuration {
      ipv4_enabled = false
      private_network = google_compute_network.default.id
      require_ssl = true
    }
    database_flags {
      name = "innodb_log_buffer_size"
      value = "536870912"
    }
    database_flags {
      name = "innodb_log_file_size"
      value = "5368709120"
    }
    database_flags {
      name = "event_scheduler"
      value = "on"
    }
    database_flags {
      name = "skip_show_database"
      value = "on"
    }
    database_flags {
      name = "local_infile"
      value = "off"
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
    cloud = "gcp"
    batch_gcp_regions = var.batch_gcp_regions
    batch_logs_bucket = module.batch_logs.name  # Deprecated
    batch_logs_storage_uri = "gs://${module.batch_logs.name}"
    hail_query_gcs_path = "gs://${module.hail_query.name}" # Deprecated
    hail_test_gcs_bucket = module.hail_test_gcs_bucket.name # Deprecated
    test_storage_uri = "gs://${module.hail_test_gcs_bucket.name}"
    query_storage_uri  = "gs://${module.hail_query.name}"
    default_namespace = "default"
    docker_root_image = local.docker_root_image
    domain = var.domain
    gcp_project = var.gcp_project
    gcp_region = var.gcp_region
    gcp_zone = var.gcp_zone
    docker_prefix = local.docker_prefix
    internal_ip = google_compute_address.internal_gateway.address
    ip = google_compute_address.gateway.address
    kubernetes_server_url = "https://${google_container_cluster.vdc.endpoint}"
    organization_domain = var.organization_domain
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
    "host": "${google_sql_database_instance.db.ip_address[0].ip_address}",
    "port": 3306,
    "user": "root",
    "password": "${random_password.db_root_password.result}",
    "instance": "${google_sql_database_instance.db.name}",
    "connection_name": "${google_sql_database_instance.db.connection_name}",
    "docker_root_image": "${local.docker_root_image}"
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

resource "google_service_account" "gcr_push" {
  account_id = "gcr-push"
  display_name = "push to gcr.io"
}

resource "google_service_account_key" "gcr_push_key" {
  service_account_id = google_service_account.gcr_push.name
}

resource "google_artifact_registry_repository_iam_member" "artifact_registry_batch_agent_viewer" {
  provider = google-beta
  project = var.gcp_project
  repository = google_artifact_registry_repository.repository.name
  location = var.gcp_location
  role = "roles/artifactregistry.reader"
  member = "serviceAccount:${google_service_account.batch_agent.email}"
}

resource "google_artifact_registry_repository_iam_member" "artifact_registry_ci_viewer" {
  provider = google-beta
  project = var.gcp_project
  repository = google_artifact_registry_repository.repository.name
  location = var.gcp_location
  role = "roles/artifactregistry.reader"
  member = "serviceAccount:${module.ci_gsa_secret.email}"
}

resource "google_storage_bucket_iam_member" "gcr_push_admin" {
  bucket = google_container_registry.registry.id
  role = "roles/storage.admin"
  member = "serviceAccount:${google_service_account.gcr_push.email}"
}

resource "google_artifact_registry_repository_iam_member" "artifact_registry_push_admin" {
  provider = google-beta
  project = var.gcp_project
  repository = google_artifact_registry_repository.repository.name
  location = var.gcp_location
  role = "roles/artifactregistry.admin"
  member = "serviceAccount:${google_service_account.gcr_push.email}"
}

# This is intended to match the secret name also used for azure credentials
# This should ultimately be replaced by using CI's own batch-managed credentials
# in BuildImage jobs
resource "kubernetes_secret" "registry_push_credentials" {
  metadata {
    name = "registry-push-credentials"
  }

  data = {
    "credentials.json" = base64decode(google_service_account_key.gcr_push_key.private_key)
  }
}

module "auth_gsa_secret" {
  source = "./gsa_k8s_secret"
  name = "auth"
  project = var.gcp_project
  iam_roles = [
    "iam.serviceAccountAdmin",
    "iam.serviceAccountKeyAdmin",
  ]
}

module "testns_auth_gsa_secret" {
  source = "./gsa_k8s_secret"
  name = "testns-auth"
  project = var.gcp_project
  iam_roles = [
    "iam.serviceAccountViewer",
  ]
}

module "batch_gsa_secret" {
  source = "./gsa_k8s_secret"
  name = "batch"
  project = var.gcp_project
  iam_roles = [
    "compute.instanceAdmin.v1",
    "iam.serviceAccountUser",
    "logging.viewer",
    "storage.admin",
  ]
}

resource "google_storage_bucket_iam_member" "batch_hail_query_bucket_storage_viewer" {
  bucket = module.hail_query.name
  role = "roles/storage.objectViewer"
  member = "serviceAccount:${module.batch_gsa_secret.email}"
}

module "testns_batch_gsa_secret" {
  source = "./gsa_k8s_secret"
  name = "testns-batch"
  project = var.gcp_project
  iam_roles = [
    "compute.instanceAdmin.v1",
    "iam.serviceAccountUser",
    "logging.viewer",
  ]
}

resource "google_storage_bucket_iam_member" "testns_batch_bucket_admin" {
  bucket = module.hail_test_gcs_bucket.name
  role = "roles/storage.admin"
  member = "serviceAccount:${module.testns_batch_gsa_secret.email}"
}

module "ci_gsa_secret" {
  source = "./gsa_k8s_secret"
  name = "ci"
  project = var.gcp_project
}

module "testns_ci_gsa_secret" {
  source = "./gsa_k8s_secret"
  name = "testns-ci"
  project = var.gcp_project
}

resource "google_storage_bucket_iam_member" "testns_ci_bucket_admin" {
  bucket = module.hail_test_gcs_bucket.name
  role = "roles/storage.admin"
  member = "serviceAccount:${module.testns_ci_gsa_secret.email}"
}

resource "google_artifact_registry_repository_iam_member" "artifact_registry_testns_ci_viewer" {
  provider = google-beta
  project = var.gcp_project
  repository = google_artifact_registry_repository.repository.name
  location = var.gcp_location
  role = "roles/artifactregistry.reader"
  member = "serviceAccount:${module.testns_ci_gsa_secret.email}"
}

resource "google_artifact_registry_repository_iam_member" "artifact_registry_viewer" {
  provider = google-beta
  project = var.gcp_project
  repository = google_artifact_registry_repository.repository.name
  location = var.gcp_location
  role = "roles/artifactregistry.reader"
  member = "serviceAccount:${module.ci_gsa_secret.email}"
}

module "monitoring_gsa_secret" {
  source = "./gsa_k8s_secret"
  name = "monitoring"
  project = var.gcp_project
}

module "grafana_gsa_secret" {
  source = "./gsa_k8s_secret"
  name = "grafana"
  project = var.gcp_project
}

module "testns_grafana_gsa_secret" {
  source = "./gsa_k8s_secret"
  name = "testns-grafana"
  project = var.gcp_project
}

# FIXME Now that there are test identities for each service, the test user no longer
# needs this many permissions. Perform an audit to see which can be removed
module "test_gsa_secret" {
  source = "./gsa_k8s_secret"
  name = "test"
  project = var.gcp_project
  iam_roles = [
    "compute.instanceAdmin.v1",
    "iam.serviceAccountUser",
    "logging.viewer",
    "serviceusage.serviceUsageConsumer",
  ]
}

resource "google_storage_bucket_iam_member" "test_bucket_admin" {
  bucket = module.hail_test_gcs_bucket.name
  role = "roles/storage.admin"
  member = "serviceAccount:${module.test_gsa_secret.email}"
}

resource "google_artifact_registry_repository_iam_member" "artifact_registry_test_viewer" {
  provider = google-beta
  project = var.gcp_project
  repository = google_artifact_registry_repository.repository.name
  location = var.gcp_location
  role = "roles/artifactregistry.reader"
  member = "serviceAccount:${module.test_gsa_secret.email}"
}

module "testns_test_gsa_secret" {
  source = "./gsa_k8s_secret"
  name = "testns-test"
  project = var.gcp_project
}

resource "google_storage_bucket_iam_member" "testns_test_gsa_bucket_admin" {
  bucket = module.hail_test_gcs_bucket.name
  role = "roles/storage.admin"
  member = "serviceAccount:${module.testns_test_gsa_secret.email}"
}

resource "google_artifact_registry_repository_iam_member" "artifact_registry_testns_test_gsa_viewer" {
  provider = google-beta
  project = var.gcp_project
  repository = google_artifact_registry_repository.repository.name
  location = var.gcp_location
  role = "roles/artifactregistry.reader"
  member = "serviceAccount:${module.testns_test_gsa_secret.email}"
}


module "test_dev_gsa_secret" {
  source = "./gsa_k8s_secret"
  name = "test-dev"
  project = var.gcp_project
}

resource "google_storage_bucket_iam_member" "test_dev_bucket_admin" {
  bucket = module.hail_test_gcs_bucket.name
  role = "roles/storage.admin"
  member = "serviceAccount:${module.test_dev_gsa_secret.email}"
}

resource "google_artifact_registry_repository_iam_member" "artifact_registry_test_dev_viewer" {
  provider = google-beta
  project = var.gcp_project
  repository = google_artifact_registry_repository.repository.name
  location = var.gcp_location
  role = "roles/artifactregistry.reader"
  member = "serviceAccount:${module.test_dev_gsa_secret.email}"
}

module "testns_test_dev_gsa_secret" {
  source = "./gsa_k8s_secret"
  name = "testns-test-dev"
  project = var.gcp_project
}

resource "google_storage_bucket_iam_member" "testns_test_dev_bucket_admin" {
  bucket = module.hail_test_gcs_bucket.name
  role = "roles/storage.admin"
  member = "serviceAccount:${module.testns_test_dev_gsa_secret.email}"
}

resource "google_artifact_registry_repository_iam_member" "artifact_registry_testns_test_dev_viewer" {
  provider = google-beta
  project = var.gcp_project
  repository = google_artifact_registry_repository.repository.name
  location = var.gcp_location
  role = "roles/artifactregistry.reader"
  member = "serviceAccount:${module.testns_test_dev_gsa_secret.email}"
}

resource "google_service_account" "batch_agent" {
  account_id = "batch2-agent"
}

resource "google_project_iam_member" "batch_agent_iam_member" {
  for_each = toset([
    "compute.instanceAdmin.v1",
    "iam.serviceAccountUser",
    "logging.logWriter",
    "storage.objectAdmin",
  ])

  project = var.gcp_project
  role = "roles/${each.key}"
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
  name = "allow-ssh"
  network = google_compute_network.default.name

  priority = 65534

  source_ranges = ["0.0.0.0/0"]

  allow {
    protocol = "tcp"
    ports = ["22"]
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

module "batch_logs" {
  source        = "./gcs_bucket"
  short_name    = "batch-logs"
  location      = var.batch_logs_bucket_location
  storage_class = var.batch_logs_bucket_storage_class
}

module "hail_query" {
  source        = "./gcs_bucket"
  short_name    = "hail-query"
  location      = var.hail_query_bucket_location
  storage_class = var.hail_query_bucket_storage_class
}

module "hail_test_gcs_bucket" {
  source        = "./gcs_bucket"
  short_name    = "hail-test"
  location      = var.hail_test_gcs_bucket_location
  storage_class = var.hail_test_gcs_bucket_storage_class
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

resource "kubernetes_cluster_role" "batch" {
  metadata {
    name = "batch"
  }

  rule {
    api_groups = [""]
    resources  = ["secrets", "serviceaccounts"]
    verbs      = ["get", "list"]
  }
}

resource "kubernetes_cluster_role_binding" "batch" {
  metadata {
    name = "batch"
  }
  role_ref {
    kind      = "ClusterRole"
    name      = "batch"
    api_group = "rbac.authorization.k8s.io"
  }
  subject {
    kind      = "ServiceAccount"
    name      = "batch"
    namespace = "default"
  }
}

data "sops_file" "auth_oauth2_client_secret_sops" {
  source_file = "${var.github_organization}/auth_oauth2_client_secret.enc.json"
}

resource "kubernetes_secret" "auth_oauth2_client_secret" {
  metadata {
    name = "auth-oauth2-client-secret"
  }

  data = {
    "client_secret.json" = data.sops_file.auth_oauth2_client_secret_sops.raw
  }
}

data "sops_file" "ci_config_sops" {
  count = fileexists("${var.github_organization}/ci_config.enc.json") ? 1 : 0
  source_file = "${var.github_organization}/ci_config.enc.json"
}

locals {
    ci_config = length(data.sops_file.ci_config_sops) == 1 ? data.sops_file.ci_config_sops[0] : null
}

module "ci" {
  source = "./ci"
  count = local.ci_config != null ? 1 : 0

  github_oauth_token = local.ci_config.data["github_oauth_token"]
  github_user1_oauth_token = local.ci_config.data["github_user1_oauth_token"]
  watched_branches = jsondecode(local.ci_config.raw).watched_branches
  deploy_steps = jsondecode(local.ci_config.raw).deploy_steps
  bucket_location = local.ci_config.data["bucket_location"]
  bucket_storage_class = local.ci_config.data["bucket_storage_class"]

  ci_email = module.ci_gsa_secret.email
  container_registry_id = google_container_registry.registry.id
  github_context = local.ci_config.data["github_context"]
}
