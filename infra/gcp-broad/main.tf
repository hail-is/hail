terraform {
  required_providers {
    google = {
      source = "hashicorp/google"
      version = "7.2.0"
    }
    google-beta = {
      source = "hashicorp/google-beta"
      version = "7.2.0"
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
  backend "gcs" {}
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
  docker_root_image = "${local.docker_prefix}/ubuntu:24.04"
}

provider "google" {
  project = var.gcp_project
  billing_project = var.gcp_project
  user_project_override = true
  region = var.gcp_region
  zone = var.gcp_zone
}

provider "google-beta" {
  project = var.gcp_project
  region = var.gcp_region
  zone = var.gcp_zone
}

data "google_client_config" "provider" {}

data "google_project" "current" {
  project_id = var.gcp_project
}

resource "google_project_service" "service_networking" {
  disable_on_destroy = false
  service = "servicenetworking.googleapis.com"
  timeouts {}
}

resource "google_compute_project_metadata" "oslogin" {
  metadata = {
    enable-oslogin = "TRUE"
    block-project-ssh-keys = "TRUE"
  }
}

# KMS Key Ring for Kubernetes secrets encryption
resource "google_kms_key_ring" "k8s_secrets" {
  name     = "k8s-secrets"
  location = var.gcp_region
}

# KMS Key for Kubernetes secrets encryption with 90-day rotation
resource "google_kms_crypto_key" "k8s_secrets_key" {
  name     = "k8s-secrets-key"
  key_ring = google_kms_key_ring.k8s_secrets.id
  purpose  = "ENCRYPT_DECRYPT"

  rotation_period = "7776000s" # 90 days in seconds

  lifecycle {
    prevent_destroy = true
  }
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
  enable_shielded_nodes = true

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

  master_authorized_networks_config {
    # Broad Institute ranges from https://ipinfo.io/AS46964
    cidr_blocks {
      cidr_block   = "69.173.64.0/18"
      display_name = "Broad Institute Range 1"
    }
    cidr_blocks {
      cidr_block   = "69.173.64.0/19"
      display_name = "Broad Institute Range 2"
    }
    cidr_blocks {
      cidr_block   = "69.173.96.0/19"
      display_name = "Broad Institute Range 3"
    }
    cidr_blocks {
      cidr_block   = "69.173.95.0/24"
      display_name = "Broad Institute Range 4"
    }
    cidr_blocks {
      cidr_block   = "69.173.97.0/24"
      display_name = "Broad Institute Range 5"
    }
    cidr_blocks {
      cidr_block   = "10.0.0.0/8"
      display_name = "VDC Internal Networks"
    }
    cidr_blocks {
      cidr_block   = "10.128.0.0/9"
      display_name = "GKE Control Plane and Nodes"
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

  timeouts {}

  addons_config {
    network_policy_config {
      disabled = false
    }
  }

  network_policy {
    enabled = true
    provider = "CALICO"
  }

  # Enable secrets encryption using CloudKMS
  database_encryption {
    state    = "ENCRYPTED"
    key_name = google_kms_crypto_key.k8s_secrets_key.id
  }

  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
  }

  control_plane_endpoints_config {
    dns_endpoint_config {
      allow_external_traffic = true
    }
  }

  workload_identity_config {
    workload_pool = "${var.gcp_project}.svc.id.goog"
  }
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
    service_account = google_service_account.gke_node_pool.email

    kubelet_config {
      cpu_manager_policy = ""
      cpu_cfs_quota = false
      pod_pids_limit = 0
    }

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

    tags = []

    shielded_instance_config {
      enable_integrity_monitoring = true
      enable_secure_boot          = true
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
    service_account = google_service_account.gke_node_pool.email

    kubelet_config {
      cpu_manager_policy = ""
      cpu_cfs_quota = false
      pod_pids_limit = 0
    }

    labels = {
      preemptible = "false"
    }

    metadata = {
      disable-legacy-endpoints = "true"
    }

    tags = []

    shielded_instance_config {
      enable_integrity_monitoring = true
      enable_secure_boot          = true
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
      ssl_mode = "TRUSTED_CLIENT_CERTIFICATE_REQUIRED"
    }

    backup_configuration {
      binary_log_enabled             = false
      enabled                        = true
      location                       = "us-central1"
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

  lifecycle {
     ignore_changes = [settings.0.tier]
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

# Cloud Router for GKE node outbound NAT
resource "google_compute_router" "gke_node_outbound_router" {
  name    = "gke-node-outbound-router"
  region  = var.gcp_region
  network = google_compute_network.default.id
}

# Cloud NAT Gateway for GKE private node outbound internet access
resource "google_compute_router_nat" "gke_node_outbound_nat" {
  name                               = "gke-node-outbound-nat"
  router                            = google_compute_router.gke_node_outbound_router.name
  region                            = var.gcp_region
  nat_ip_allocate_option            = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"

  log_config {
    enable = true
    filter = "ERRORS_ONLY"
  }
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
  cleanup_policy_dry_run = false

  # https://github.com/hashicorp/terraform-provider-azurerm/issues/7396
  lifecycle {
    ignore_changes = [cleanup_policies, timeouts]
  }
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

resource "google_artifact_registry_repository_iam_member" "artifact_registry_ci_admin" {
  provider = google-beta
  project = var.gcp_project
  repository = google_artifact_registry_repository.repository.name
  location = var.artifact_registry_location
  role = "roles/artifactregistry.admin"
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

resource "google_project_iam_custom_role" "auth_service_account_manager" {
  role_id     = "authServiceAccountManager"
  title       = "Auth Service Account Manager"
  description = "Custom role for auth service with minimal required permissions for service account management"

  permissions = [
    "iam.serviceAccounts.create",
    "iam.serviceAccounts.delete",
    "iam.serviceAccounts.get",
    "iam.serviceAccounts.list",
    "iam.serviceAccounts.setIamPolicy",
    "iam.serviceAccountKeys.create",
  ]
}

module "auth_gsa_secret" {
  source = "./gsa"
  name = "auth"
  project = var.gcp_project
  iam_roles = [
    "projects/${var.gcp_project}/roles/authServiceAccountManager",
    "cloudprofiler.agent",
  ]
}

module "testns_auth_gsa_secret" {
  source = "./gsa"
  name = "testns-auth"
  project = var.gcp_project
  iam_roles = [
    "iam.serviceAccountViewer",
    "cloudprofiler.agent",
  ]
}

resource "google_project_iam_custom_role" "batch_compute_manager" {
  role_id     = "batchComputeManager"
  title       = "Batch Compute Manager"
  description = "Custom role for batch service with specific compute permissions for instance management"

  permissions = [
    "compute.disks.create",
    "compute.disks.delete",
    "compute.disks.list",
    "compute.images.useReadOnly",
    "compute.instances.create",
    "compute.instances.delete",
    "compute.instances.get",
    "compute.instances.list",
    "compute.instances.setLabels",
    "compute.instances.setMetadata",
    "compute.instances.setServiceAccount",
    "compute.instances.setTags",
    "compute.machineTypes.list",
    "compute.regions.get",
    "compute.subnetworks.use",
    "compute.subnetworks.useExternalIp",
    "compute.zoneOperations.get",
  ]
}

module "batch_gsa_secret" {
  source = "./gsa"
  name = "batch"
  project = var.gcp_project
  iam_roles = [
    "projects/${var.gcp_project}/roles/batchComputeManager",
    "logging.viewer",
    "cloudprofiler.agent",
  ]
}

resource "google_storage_bucket_iam_member" "batch_hail_query_bucket_storage_viewer" {
  bucket = google_storage_bucket.hail_query.name
  role = "roles/storage.objectViewer"
  member = "serviceAccount:${module.batch_gsa_secret.email}"
}

module "testns_batch_gsa_secret" {
  source = "./gsa"
  name = "testns-batch"
  project = var.gcp_project
  iam_roles = [
    "projects/${var.gcp_project}/roles/batchComputeManager",
    "logging.viewer",
    "cloudprofiler.agent",
  ]
}

resource "google_storage_bucket_iam_member" "testns_batch_bucket_admin" {
  bucket = google_storage_bucket.hail_test_bucket.name
  role = "roles/storage.admin"
  member = "serviceAccount:${module.testns_batch_gsa_secret.email}"
}

module "ci_gsa_secret" {
  source = "./gsa"
  name = "ci"
  project = var.gcp_project
  iam_roles = [
    "cloudprofiler.agent",
  ]
}

module "testns_ci_gsa_secret" {
  source = "./gsa"
  name = "testns-ci"
  project = var.gcp_project
}

resource "google_storage_bucket_iam_member" "testns_ci_bucket_admin" {
  bucket = google_storage_bucket.hail_test_bucket.name
  role = "roles/storage.admin"
  member = "serviceAccount:${module.testns_ci_gsa_secret.email}"
}

resource "google_artifact_registry_repository_iam_member" "artifact_registry_testns_ci_repo_admin" {
  provider = google-beta
  project = var.gcp_project
  repository = google_artifact_registry_repository.repository.name
  location = var.artifact_registry_location
  role = "roles/artifactregistry.repoAdmin"
  member = "serviceAccount:${module.testns_ci_gsa_secret.email}"
}

module "grafana_gsa_secret" {
  source = "./gsa"
  name = "grafana"
  project = var.gcp_project
  iam_roles = [
    "monitoring.viewer",
  ]
}

module "testns_grafana_gsa_secret" {
  source = "./gsa"
  name = "testns-grafana"
  project = var.gcp_project
  iam_roles = [
    "monitoring.viewer",
  ]
}

module "test_gsa_secret" {
  source = "./gsa"
  name = "test"
  project = var.gcp_project
  iam_roles = [
    "serviceusage.serviceUsageConsumer",
  ]
}

resource "google_storage_bucket_iam_member" "test_bucket_admin" {
  bucket = google_storage_bucket.hail_test_bucket.name
  role = "roles/storage.admin"
  member = "serviceAccount:${module.test_gsa_secret.email}"
}

resource "google_storage_bucket_iam_member" "test_requester_pays_bucket_admin" {
  bucket = google_storage_bucket.hail_test_requester_pays_bucket.name
  role = "roles/storage.admin"
  member = "serviceAccount:${module.test_gsa_secret.email}"
}

resource "google_artifact_registry_repository_iam_member" "artifact_registry_test_gsa_viewer" {
  provider = google-beta
  project = var.gcp_project
  repository = google_artifact_registry_repository.repository.name
  location = var.artifact_registry_location
  role = "roles/artifactregistry.reader"
  member = "serviceAccount:${module.test_gsa_secret.email}"
}

module "testns_test_gsa_secret" {
  source = "./gsa"
  name = "testns-test"
  project = var.gcp_project
  iam_roles = [
    "serviceusage.serviceUsageConsumer",
  ]
}

resource "google_storage_bucket_iam_member" "testns_test_gsa_bucket_admin" {
  bucket = google_storage_bucket.hail_test_bucket.name
  role = "roles/storage.admin"
  member = "serviceAccount:${module.testns_test_gsa_secret.email}"
}

resource "google_storage_bucket_iam_member" "testns_test_gsa_requester_pays_bucket_admin" {
  bucket = google_storage_bucket.hail_test_requester_pays_bucket.name
  role = "roles/storage.admin"
  member = "serviceAccount:${module.testns_test_gsa_secret.email}"
}

resource "google_artifact_registry_repository_iam_member" "artifact_registry_testns_test_gsa_viewer" {
  provider = google-beta
  project = var.gcp_project
  repository = google_artifact_registry_repository.repository.name
  location = var.artifact_registry_location
  role = "roles/artifactregistry.reader"
  member = "serviceAccount:${module.testns_test_gsa_secret.email}"
}

module "test_dev_gsa_secret" {
  source = "./gsa"
  name = "test-dev"
  project = var.gcp_project
}

resource "google_storage_bucket_iam_member" "test_dev_bucket_admin" {
  bucket = google_storage_bucket.hail_test_bucket.name
  role = "roles/storage.admin"
  member = "serviceAccount:${module.test_dev_gsa_secret.email}"
}

resource "google_artifact_registry_repository_iam_member" "artifact_registry_test_dev_viewer" {
  provider = google-beta
  project = var.gcp_project
  repository = google_artifact_registry_repository.repository.name
  location = var.artifact_registry_location
  role = "roles/artifactregistry.reader"
  member = "serviceAccount:${module.test_dev_gsa_secret.email}"
}

module "testns_test_dev_gsa_secret" {
  source = "./gsa"
  name = "testns-test-dev"
  project = var.gcp_project
}

resource "google_storage_bucket_iam_member" "testns_test_dev_bucket_admin" {
  bucket = google_storage_bucket.hail_test_bucket.name
  role = "roles/storage.admin"
  member = "serviceAccount:${module.testns_test_dev_gsa_secret.email}"
}

resource "google_artifact_registry_repository_iam_member" "artifact_registry_testns_test_dev_viewer" {
  provider = google-beta
  project = var.gcp_project
  repository = google_artifact_registry_repository.repository.name
  location = var.artifact_registry_location
  role = "roles/artifactregistry.reader"
  member = "serviceAccount:${module.testns_test_dev_gsa_secret.email}"
}

resource "google_project_iam_custom_role" "batch2_agent_compute_ops" {
  role_id     = "batch2AgentComputeOps"
  title       = "Batch2 Agent Compute Ops"
  description = "Custom role for batch2-agent with minimal disk and instance management permissions"

  permissions = [
    "compute.disks.create",
    "compute.disks.delete",
    "compute.disks.setLabels",
    "compute.disks.use",
    "compute.instances.attachDisk",
    "compute.instances.delete",
    "compute.instances.detachDisk",
    "compute.zoneOperations.get",
  ]
}

resource "google_service_account" "batch_agent" {
  description  = "Delete instances and pull images"
  display_name = "batch2-agent"
  account_id = "batch2-agent"
}

resource "google_project_iam_member" "batch_agent_iam_member" {
  for_each = toset([
    "logging.logWriter",
    "storage.objectCreator",
    "storage.objectViewer",
  ])

  project = var.gcp_project
  role = "roles/${each.key}"
  member = "serviceAccount:${google_service_account.batch_agent.email}"
}

resource "google_project_iam_member" "batch_agent_custom_role" {
  project = var.gcp_project
  role = "projects/${var.gcp_project}/roles/batch2AgentComputeOps"
  member = "serviceAccount:${google_service_account.batch_agent.email}"
}

# Grant batch service account permission to act as batch2-agent service account
resource "google_service_account_iam_member" "batch_act_as_batch_agent" {
  service_account_id = google_service_account.batch_agent.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${module.batch_gsa_secret.email}"
}

# Grant testns-batch service account permission to act as batch2-agent service account
resource "google_service_account_iam_member" "testns_batch_act_as_batch_agent" {
  service_account_id = google_service_account.batch_agent.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${module.testns_batch_gsa_secret.email}"
}

# Grant batch2-agent service account permission to act as itself - required for some permission checks
resource "google_service_account_iam_member" "batch_agent_act_as_batch_agent" {
  service_account_id = google_service_account.batch_agent.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.batch_agent.email}"
}

resource "google_service_account" "gke_node_pool" {
  description  = "Service account for GKE node pools"
  display_name = "gke-node-pool"
  account_id = "gke-node-pool"
}

resource "google_project_iam_member" "gke_node_pool_iam_member" {
  for_each = toset([
    "storage.objectViewer",
    "logging.logWriter",
    "monitoring.metricWriter",
    "monitoring.viewer",
    "autoscaling.metricsWriter",
    "artifactregistry.reader",
  ])

  project = var.gcp_project
  role = "roles/${each.key}"
  member = "serviceAccount:${google_service_account.gke_node_pool.email}"
}

# Grant CloudKMS permissions to GKE service account for secrets encryption
resource "google_kms_crypto_key_iam_binding" "gke_service_account_kms_encrypter_decrypter" {
  crypto_key_id = google_kms_crypto_key.k8s_secrets_key.id
  role          = "roles/cloudkms.cryptoKeyEncrypterDecrypter"

  members = [
    "serviceAccount:service-${data.google_project.current.number}@container-engine-robot.iam.gserviceaccount.com",
  ]
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

  labels = {
    "name" = "hail-test-requester-pays-fds32"
  }

  timeouts {}
}

resource "google_dns_managed_zone" "dns_zone" {
  description = "hail managed dns zone"
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

resource "kubernetes_pod_disruption_budget_v1" "kube_dns_pdb" {
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

resource "kubernetes_pod_disruption_budget_v1" "kube_dns_autoscaler_pdb" {
  metadata {
    name = "kube-dns-autoscaler"
    namespace = "kube-system"
  }
  spec {
    max_unavailable = "1"
    selector {
      match_labels = {
        k8s-app = "kube-dns-autoscaler"
      }
    }
  }
}

resource "kubernetes_pod_disruption_budget_v1" "event_exporter_pdb" {
  metadata {
    name = "event-exporter"
    namespace = "kube-system"
  }
  spec {
    max_unavailable = "1"
    selector {
      match_labels = {
	# nb: pods are called event-exporter-gke-...
        k8s-app = "event-exporter"
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

  github_organization = var.github_organization
}
