terraform {
  required_providers {
    google = {
      source = "hashicorp/google"
      # Updated 2024-08-08 to avoid race conditions during rapid service account creation
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

variable "enable_master_authorized_networks" {
  type = bool
  description = "Enable master authorized networks configuration for GKE cluster"
  default = false
}

variable "master_authorized_networks" {
  type = list(object({
    cidr_block   = string
    display_name = string
  }))
  description = "List of CIDR blocks authorized to access the GKE master"
  default = []
}

locals {
  docker_prefix = (
    var.use_artifact_registry ?
    "${var.gcp_region}-docker.pkg.dev/${var.gcp_project}/hail" :
    "gcr.io/${var.gcp_project}"
  )
  docker_root_image = "${local.docker_prefix}/ubuntu:24.04"
  dockerhub_prefix  = "${var.gcp_location}-docker.pkg.dev/${var.gcp_project}/dockerhubproxy"
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

data "google_project" "current" {
  project_id = var.gcp_project
}

resource "google_project_service" "service_networking" {
  service = "servicenetworking.googleapis.com"
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
  enable_shielded_nodes = true

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

  dynamic "master_authorized_networks_config" {
    for_each = var.enable_master_authorized_networks && length(var.master_authorized_networks) > 0 ? [1] : []
    content {
      dynamic "cidr_blocks" {
        for_each = var.master_authorized_networks
        content {
          cidr_block   = cidr_blocks.value.cidr_block
          display_name = cidr_blocks.value.display_name
        }
      }
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
  initial_node_count = 1

  autoscaling {
    min_node_count = 0
    max_node_count = 200
  }

  node_config {
    spot = true
    machine_type = "n1-standard-4"
    service_account = google_service_account.gke_node_pool.email

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

    shielded_instance_config {
      enable_integrity_monitoring = true
      enable_secure_boot          = true
    }

    workload_metadata_config {
      mode = "GKE_METADATA"
    }
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
    service_account = google_service_account.gke_node_pool.email

    labels = {
      preemptible = "false"
    }

    metadata = {
      disable-legacy-endpoints = "true"
    }

    shielded_instance_config {
      enable_integrity_monitoring = true
      enable_secure_boot          = true
    }

    workload_metadata_config {
      mode = "GKE_METADATA"
    }
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
      ssl_mode = "TRUSTED_CLIENT_CERTIFICATE_REQUIRED"
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

# Parse batch_gcp_regions to create Cloud NAT for all regions where batch instances can be created
locals {
  batch_regions = jsondecode(var.batch_gcp_regions)
}

# Cloud Routers for outbound NAT in all batch regions
resource "google_compute_router" "outbound_router" {
  for_each = toset(local.batch_regions)
  name     = "outbound-router-${each.key}"
  region   = each.key
  network  = google_compute_network.default.id
}

# Cloud NAT Gateways for outbound internet access in all batch regions
resource "google_compute_router_nat" "outbound_nat" {
  for_each = google_compute_router.outbound_router
  name                               = "outbound-nat-${each.key}"
  router                            = each.value.name
  region                            = each.key
  nat_ip_allocate_option            = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"
  enable_dynamic_port_allocation     = true
  min_ports_per_vm                  = 512
  max_ports_per_vm                  = 16384

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
    dockerhub_prefix = local.dockerhub_prefix
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
  # Note: As of August 8 2024, *new* "container registry" instances get converted into "artifact repository" instances
  # and the registry cannot be found after 'creation'. GCR is scheduled for a general switch-off in March 2025
  # so this is only left here for backwards compatibility until then.
  count = var.use_artifact_registry ? 0 : 1
}

resource "google_artifact_registry_repository" "repository" {
  provider = google-beta
  format = "DOCKER"
  repository_id = "hail"
  location = var.gcp_location
}

resource "google_artifact_registry_repository" "dockerhub_remote" {
  provider      = google-beta
  format        = "DOCKER"
  mode          = "REMOTE"
  repository_id = "dockerhubproxy"
  location      = var.gcp_location

  remote_repository_config {
    description = "Docker Hub remote repository for Batch worker images"
    docker_repository {
      public_repository = "DOCKER_HUB"
    }
  }
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

# Grant all service accounts in the project read access to dockerhubproxy
resource "google_artifact_registry_repository_iam_member" "artifact_registry_all_service_accounts_dockerhub_viewer" {
  provider  = google-beta
  project   = var.gcp_project
  repository = google_artifact_registry_repository.dockerhub_remote.name
  location  = var.gcp_location
  role      = "roles/artifactregistry.reader"
  member    = "principalSet://cloudresourcemanager.googleapis.com/projects/${data.google_project.current.number}/type/ServiceAccount"
}


resource "google_artifact_registry_repository_iam_member" "artifact_registry_ci_admin" {
  provider = google-beta
  project = var.gcp_project
  repository = google_artifact_registry_repository.repository.name
  location = var.gcp_location
  role = "roles/artifactregistry.admin"
  member = "serviceAccount:${module.ci_gsa_secret.email}"
}

resource "google_storage_bucket_iam_member" "gcr_push_admin" {
  count = var.use_artifact_registry ? 0 : 1
  bucket = google_container_registry.registry[0].id
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
  source = "./gsa_k8s_secret"
  name = "auth"
  project = var.gcp_project
  iam_roles = [
    "projects/${var.gcp_project}/roles/authServiceAccountManager",
    "cloudprofiler.agent",
  ]
}

module "testns_auth_gsa_secret" {
  source = "./gsa_k8s_secret"
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
  source = "./gsa_k8s_secret"
  name = "batch"
  project = var.gcp_project
  iam_roles = [
    "projects/${var.gcp_project}/roles/batchComputeManager",
    "logging.viewer",
    "storage.admin",
    "cloudprofiler.agent",
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
    "projects/${var.gcp_project}/roles/batchComputeManager",
    "logging.viewer",
    "cloudprofiler.agent",
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
  iam_roles = [
    "cloudprofiler.agent",
  ]
}

module "testns_ci_gsa_secret" {
  source = "./gsa_k8s_secret"
  name = "testns-ci"
  project = var.gcp_project
  iam_roles = [
    "cloudprofiler.agent",
  ]
}

resource "google_storage_bucket_iam_member" "testns_ci_bucket_admin" {
  bucket = module.hail_test_gcs_bucket.name
  role = "roles/storage.admin"
  member = "serviceAccount:${module.testns_ci_gsa_secret.email}"
}

resource "google_artifact_registry_repository_iam_member" "artifact_registry_testns_ci_repo_admin" {
  provider = google-beta
  project = var.gcp_project
  repository = google_artifact_registry_repository.repository.name
  location = var.gcp_location
  role = "roles/artifactregistry.repoAdmin"
  member = "serviceAccount:${module.testns_ci_gsa_secret.email}"
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
  iam_roles = [
    "monitoring.viewer",
  ]
}

module "testns_grafana_gsa_secret" {
  source = "./gsa_k8s_secret"
  name = "testns-grafana"
  project = var.gcp_project
  iam_roles = [
    "monitoring.viewer",
  ]
}

module "test_gsa_secret" {
  source = "./gsa_k8s_secret"
  name = "test"
  project = var.gcp_project
  iam_roles = [
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
  account_id = "batch2-agent"
}

resource "google_project_iam_member" "batch_agent_iam_member" {
  for_each = toset([
    "logging.logWriter",
    "storage.objectAdmin",
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

data "sops_file" "hailctl_client_secret_sops" {
  source_file = "${var.github_organization}/hailctl_client_secret.enc.json"
}

resource "kubernetes_secret" "auth_oauth2_client_secret" {
  metadata {
    name = "auth-oauth2-client-secret"
  }

  data = {
    "client_secret.json" = data.sops_file.auth_oauth2_client_secret_sops.raw
    "hailctl_client_secret.json" = data.sops_file.hailctl_client_secret_sops.raw
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
  # For now, GCP CI via this terraform relies on container registry, which is going away it March 2025
  # so this might need adapting to be more like the gcp-broad main.tf when that happens
  container_registry_id = google_container_registry.registry[0].id
  github_context = local.ci_config.data["github_context"]
}
