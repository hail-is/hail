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
variable artifact_registry_location {}

variable deploy_ukbb {
  type = bool
  description = "Run the UKBB Genetic Correlation browser"
  default = false
}
variable default_subnet_ip_cidr_range {}

locals {
  docker_prefix = (
    var.use_artifact_registry ?
    "${var.gcp_region}-docker.pkg.dev/${var.gcp_project}/hail" :
    "gcr.io/${var.gcp_project}"
  )
  docker_root_image = "${local.docker_prefix}/ubuntu:20.04"
}

provider "google" {
  project = var.gcp_project
  region = var.gcp_region
  zone = var.gcp_zone
}

provider "google-beta" {
  project = var.gcp_project
  region = var.gcp_region
  zone = var.gcp_zone
}

data "google_client_config" "provider" {}

resource "google_project_service" "service_networking" {
  disable_on_destroy = false
  service = "servicenetworking.googleapis.com"
  timeouts {}
}

resource "google_compute_network" "default" {
  name = "default"
  description = "Default network for the project"
  enable_ula_internal_ipv6 = false
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "default_region" {
  name = "default"
  region = var.gcp_region
  network = google_compute_network.default.id
  ip_cidr_range = var.default_subnet_ip_cidr_range
  private_ip_google_access = true

  log_config {
    aggregation_interval = "INTERVAL_15_MIN"
    flow_sampling        = 0.5
    metadata             = "EXCLUDE_ALL_METADATA"
  }

  timeouts {}
}

resource "google_compute_subnetwork" "us_nondefault_subnets" {
  for_each = {
    us-east1 = "10.142.0.0/20",
    us-east4 = "10.150.0.0/20",
    us-east5 = "10.202.0.0/20",
    us-east7 = "10.196.0.0/20",
    us-south1 = "10.206.0.0/20",
    us-west1 = "10.138.0.0/20",
    us-west2 = "10.168.0.0/20",
    us-west3 = "10.180.0.0/20",
    us-west4 = "10.182.0.0/20",
  }

  name = "default"
  region = each.key
  network = google_compute_network.default.id
  ip_cidr_range = each.value
  private_ip_google_access = true

  log_config {
    aggregation_interval = "INTERVAL_15_MIN"
    flow_sampling        = 0.5
    metadata             = "EXCLUDE_ALL_METADATA"
  }

  timeouts {}
}

resource "google_container_cluster" "vdc" {
  provider = google-beta
  name = "vdc"
  location = var.gcp_zone
  network = google_compute_network.default.name
  enable_shielded_nodes = false

  # We can't create a cluster with no node pool defined, but we want to only use
  # separately managed node pools. So we create the smallest possible default
  # node pool and immediately delete it.
  # remove_default_node_pool = true
  remove_default_node_pool = null
  initial_node_count = 0

  resource_labels = {
    role = "vdc"
  }

  release_channel {
    channel = "REGULAR"
  }

  master_auth {
    client_certificate_config {
      issue_client_certificate = false
    }
  }

  cluster_autoscaling {
    # Don't use node auto-provisioning since we manage node pools ourselves
    enabled = false
    autoscaling_profile = "OPTIMIZE_UTILIZATION"
  }

  resource_usage_export_config {
    enable_network_egress_metering       = false
    enable_resource_consumption_metering = true

    bigquery_destination {
      dataset_id = "gke_vdc_usage"
    }
  }

  workload_identity_config {
    workload_pool = "hail-vdc.svc.id.goog"
  }

  timeouts {}
}

resource "google_container_node_pool" "vdc_preemptible_pool" {
  name     = var.k8s_preemptible_node_pool_name
  location = var.gcp_zone
  cluster  = google_container_cluster.vdc.name

  # Allocate at least one node, so that autoscaling can take place.
  initial_node_count = 3

  autoscaling {
    min_node_count = 0
    max_node_count = 200
  }

  node_config {
    spot = true
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
      "https://www.googleapis.com/auth/devstorage.read_only",
      "https://www.googleapis.com/auth/logging.write",
      "https://www.googleapis.com/auth/monitoring",
      "https://www.googleapis.com/auth/service.management.readonly",
      "https://www.googleapis.com/auth/servicecontrol",
      "https://www.googleapis.com/auth/trace.append",
    ]
    tags = []

    shielded_instance_config {
      enable_integrity_monitoring = true
      enable_secure_boot          = false
    }

    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }

  timeouts {}

  upgrade_settings {
    max_surge       = 1
    max_unavailable = 0
  }
}

resource "google_container_node_pool" "vdc_nonpreemptible_pool" {
  name     = var.k8s_nonpreemptible_node_pool_name
  location = var.gcp_zone
  cluster  = google_container_cluster.vdc.name

  # Allocate at least one node, so that autoscaling can take place.
  initial_node_count = 2

  autoscaling {
    min_node_count = 0
    max_node_count = 200
  }

  management {
    auto_repair  = true
    auto_upgrade = true
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
      "https://www.googleapis.com/auth/devstorage.read_only",
      "https://www.googleapis.com/auth/logging.write",
      "https://www.googleapis.com/auth/monitoring",
      "https://www.googleapis.com/auth/service.management.readonly",
      "https://www.googleapis.com/auth/servicecontrol",
      "https://www.googleapis.com/auth/trace.append",
    ]

    tags = []

    shielded_instance_config {
      enable_integrity_monitoring = true
      enable_secure_boot          = false
    }

    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }

  timeouts {}

  upgrade_settings {
    max_surge       = 1
    max_unavailable = 0
  }
}

resource "random_string" "db_name_suffix" {
  length = 5
  special = false
  numeric = false
  upper = false
  lifecycle {
    ignore_changes = [
      special,
      numeric,
      upper,
    ]
  }
}

resource "google_service_networking_connection" "private_vpc_connection" {
  # google_compute_network returns the name as the project but our extant networking connection uses
  # the number
  network = "projects/859893752941/global/networks/default" # google_compute_network.default.id
  service = "services/servicenetworking.googleapis.com"
  reserved_peering_ranges = [
    "jg-test-clone-resource-id-ip-range",
  ]

  timeouts {}
}


resource "google_sql_database_instance" "db" {
  name = "db-${random_string.db_name_suffix.result}"
  database_version = "MYSQL_8_0_28"
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

    backup_configuration {
      binary_log_enabled             = false
      enabled                        = true
      location                       = "us"
      point_in_time_recovery_enabled = false
      start_time                     = "13:00"
      transaction_log_retention_days = 7

      backup_retention_settings {
	retained_backups = 7
	retention_unit   = "COUNT"
      }
    }

    database_flags {
      name  = "innodb_log_buffer_size"
      value = "536870912"
    }
    database_flags {
      name  = "innodb_log_file_size"
      value = "5368709120"
    }
    database_flags {
      name  = "event_scheduler"
      value = "on"
    }
    database_flags {
      name  = "skip_show_database"
      value = "on"
    }
    database_flags {
      name  = "local_infile"
      value = "off"
    }

    insights_config {
      query_insights_enabled  = true
      query_string_length     = 1024
      record_application_tags = false
      record_client_address   = false
    }

    location_preference {
      zone = "us-central1-a"
    }

    maintenance_window {
      day  = 7
      hour = 16
    }
  }

  timeouts {}
}

resource "google_compute_address" "gateway" {
  name = "site"
  region = var.gcp_region
}

resource "google_compute_address" "internal_gateway" {
  name = "internal-gateway"
  # subnetwork = data.google_compute_subnetwork.default_region.id
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

resource "google_artifact_registry_repository" "repository" {
  provider = google-beta
  format = "DOCKER"
  repository_id = "hail"
  location = var.artifact_registry_location
}

resource "google_service_account" "gcr_push" {
  account_id = "gcr-push"
  display_name = "push to gcr.io"
}

resource "google_artifact_registry_repository_iam_member" "artifact_registry_batch_agent_viewer" {
  provider = google-beta
  project = var.gcp_project
  repository = google_artifact_registry_repository.repository.name
  location = var.artifact_registry_location
  role = "roles/artifactregistry.reader"
  member = "serviceAccount:${google_service_account.batch_agent.email}"
}

resource "google_artifact_registry_repository_iam_member" "artifact_registry_ci_viewer" {
  provider = google-beta
  project = var.gcp_project
  repository = google_artifact_registry_repository.repository.name
  location = var.artifact_registry_location
  role = "roles/artifactregistry.reader"
  member = "serviceAccount:${module.ci_gsa_secret.email}"
}

resource "google_artifact_registry_repository_iam_member" "artifact_registry_push_admin" {
  provider = google-beta
  project = var.gcp_project
  repository = google_artifact_registry_repository.repository.name
  location = var.artifact_registry_location
  role = "roles/artifactregistry.repoAdmin"
  member = "serviceAccount:${google_service_account.gcr_push.email}"
}

module "ukbb" {
  count = var.deploy_ukbb ? 1 : 0
  source = "../k8s/ukbb"
}

module "auth_gsa_secret" {
  source = "./gsa"
  name = "auth"
  project = var.gcp_project
  iam_roles = [
    "iam.serviceAccountAdmin",
    "iam.serviceAccountKeyAdmin",
  ]
}

module "batch_gsa_secret" {
  source = "./gsa"
  name = "batch"
  project = var.gcp_project
  iam_roles = [
    "compute.instanceAdmin.v1",
    "iam.serviceAccountUser",
    "logging.viewer",
  ]
}

resource "google_storage_bucket_iam_member" "batch_hail_query_bucket_storage_viewer" {
  bucket = google_storage_bucket.hail_query.name
  role = "roles/storage.objectViewer"
  member = "serviceAccount:${module.batch_gsa_secret.email}"
}

module "ci_gsa_secret" {
  source = "./gsa"
  name = "ci"
  project = var.gcp_project
}

resource "google_artifact_registry_repository_iam_member" "artifact_registry_viewer" {
  provider = google-beta
  project = var.gcp_project
  repository = google_artifact_registry_repository.repository.name
  location = var.artifact_registry_location
  role = "roles/artifactregistry.reader"
  member = "serviceAccount:${module.ci_gsa_secret.email}"
}

module "grafana_gsa_secret" {
  source = "./gsa"
  name = "grafana"
  project = var.gcp_project
}

module "test_gsa_secret" {
  source = "./gsa"
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
  bucket = google_storage_bucket.hail_test_bucket.name
  role = "roles/storage.admin"
  member = "serviceAccount:${module.test_gsa_secret.email}"
}

resource "google_service_account" "batch_agent" {
  description  = "Delete instances and pull images"
  display_name = "batch2-agent"
  account_id = "batch2-agent"
}

resource "google_project_iam_member" "batch_agent_iam_member" {
  for_each = toset([
    "compute.instanceAdmin.v1",
    "iam.serviceAccountUser",
    "logging.logWriter",
    "storage.objectCreator",
    "storage.objectViewer",
  ])

  project = var.gcp_project
  role = "roles/${each.key}"
  member = "serviceAccount:${google_service_account.batch_agent.email}"
}

resource "google_compute_firewall" "default_allow_internal" {
  name    = "default-allow-internal"
  network = google_compute_network.default.name

  priority = 1000

  source_ranges = ["10.128.0.0/9"]

  allow {
    ports    = []
    protocol = "all"
  }

  timeouts {}
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

resource "google_storage_bucket" "batch_logs" {
  name = "hail-batch"
  location = var.batch_logs_bucket_location
  storage_class = var.batch_logs_bucket_storage_class
  uniform_bucket_level_access = true
  labels = {
    "name" = "hail-batch"
  }
  timeouts {}
}


resource "random_string" "hail_query_bucket_suffix" {
  length = 5
}

resource "google_storage_bucket" "hail_query" {
  name = "hail-query-${random_string.hail_query_bucket_suffix.result}"
  location = var.hail_query_bucket_location
  storage_class = var.hail_query_bucket_storage_class
  uniform_bucket_level_access = true
  labels = {
    "name" = "hail-query-${random_string.hail_query_bucket_suffix.result}"
  }
  timeouts {}
}

resource "random_string" "hail_test_bucket_suffix" {
  length = 5
}

resource "google_storage_bucket" "hail_test_bucket" {
  name = "hail-test-${random_string.hail_test_bucket_suffix.result}"
  location = var.hail_test_gcs_bucket_location
  force_destroy = false
  storage_class = var.hail_test_gcs_bucket_storage_class
  uniform_bucket_level_access = true
  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      age                        = 1
      days_since_custom_time     = 0
      days_since_noncurrent_time = 0
      matches_prefix             = []
      matches_storage_class      = []
      matches_suffix             = []
      num_newer_versions         = 0
      with_state                 = "ANY"
    }
  }

  timeouts {}
}

resource "random_string" "hail_test_requester_pays_bucket_suffix" {
  length = 5
}

resource "google_storage_bucket" "hail_test_requester_pays_bucket" {
  name = "hail-test-requester-pays-${random_string.hail_test_requester_pays_bucket_suffix.result}"
  location = var.hail_test_gcs_bucket_location
  force_destroy = false
  storage_class = var.hail_test_gcs_bucket_storage_class
  uniform_bucket_level_access = true
  requester_pays = true

  timeouts {}
}

resource "google_dns_managed_zone" "dns_zone" {
  description = ""
  name = "hail"
  dns_name = "hail."
  visibility = "private"

  private_visibility_config {
    networks {
      network_url = google_compute_network.default.self_link
    }
  }

  timeouts {}
}

resource "google_dns_record_set" "internal_gateway" {
  name = "*.${google_dns_managed_zone.dns_zone.dns_name}"
  managed_zone = google_dns_managed_zone.dns_zone.name
  type = "A"
  ttl = 3600

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

resource "kubernetes_pod_disruption_budget" "kube_dns_pdb" {
  metadata {
    name = "kube-dns"
    namespace = "kube-system"
  }
  spec {
    max_unavailable = "1"
    selector {
      match_labels = {
        k8s-app = "kube-dns"
      }
    }
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
  github_context = local.ci_config.data["github_context"]
  test_oauth2_callback_urls = local.ci_config.data["test_oauth2_callback_urls"]
}
