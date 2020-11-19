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

variable "gcp_project" {}
variable "gcp_region" {}
variable "gcp_zone" {}
variable "domain" {}

provider "google" {
  credentials = file("~/.hail/terraform_sa_key.json")

  project = var.gcp_project
  region  = var.gcp_region
  zone    = var.gcp_zone
}

data "google_client_config" "provider" {}

resource "google_project_service" "service_networking" {
  service = "servicenetworking.googleapis.com"
}

resource "google_compute_network" "internal" {
  name = "internal"
}

data "google_compute_subnetwork" "internal_default_region" {
  name = "internal"
  region = var.gcp_region
  depends_on = [google_compute_network.internal]
}

resource "google_container_cluster" "vdc" {
  name = "vdc"
  location = var.gcp_zone
  network = google_compute_network.internal.name

  # We can't create a cluster with no node pool defined, but we want to only use
  # separately managed node pools. So we create the smallest possible default
  # node pool and immediately delete it.
  remove_default_node_pool = true
  initial_node_count       = 1

  master_auth {
    username = ""
    password = ""

    client_certificate_config {
      issue_client_certificate = false
    }
  }
}

resource "google_container_node_pool" "vdc_preemptible_pool" {
  name       = "preemptible-pool"
  location   = var.gcp_zone
  cluster    = google_container_cluster.vdc.name

  autoscaling {
    min_node_count = 0
    max_node_count = 200
  }

  node_config {
    preemptible  = true
    machine_type = "n1-standard-2"

    metadata = {
      disable-legacy-endpoints = "true"
    }

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
  }
}

resource "google_container_node_pool" "vdc_nonpreemptible_pool" {
  name       = "nonpreemptible-pool"
  location   = var.gcp_zone
  cluster    = google_container_cluster.vdc.name

  autoscaling {
    min_node_count = 0
    max_node_count = 200
  }

  node_config {
    preemptible  = false
    machine_type = "n1-standard-2"

    metadata = {
      disable-legacy-endpoints = "true"
    }

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
  }
}

resource "google_compute_global_address" "db_ip_address" {
  name          = "db-ip-address"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network = google_compute_network.internal.id
}

resource "google_service_networking_connection" "private_vpc_connection" {
  network = google_compute_network.internal.id
  service = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.db_ip_address.name]
}

resource "random_id" "db_name_suffix" {
  byte_length = 4
}

resource "google_sql_database_instance" "db" {
  name             = "db-${random_id.db_name_suffix.hex}"
  database_version = "MYSQL_5_7"
  region           = var.gcp_region

  depends_on = [google_service_networking_connection.private_vpc_connection]

  settings {
    # Second-generation instance tiers are based on the machine
    # type. See argument reference below.
    tier = "db-n1-standard-1"

    ip_configuration {
      ipv4_enabled    = false
      private_network = google_compute_network.internal.id
      require_ssl = true
    }
  }
}

resource "google_compute_global_address" "gateway" {
  name = "gateway"
}

resource "google_compute_address" "internal_gateway" {
  name         = "internal-gateway"
  subnetwork   = data.google_compute_subnetwork.internal_default_region.id
  address_type = "INTERNAL"
  region       = var.gcp_region
}

provider "kubernetes" {
  load_config_file = false

  host  = "https://${google_container_cluster.vdc.endpoint}"
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
    gcp_project = var.gcp_project
    domain = var.domain
    internal_ip = google_compute_address.internal_gateway.address
    ip = google_compute_global_address.gateway.address
    kubernetes_server_url = "https://${google_container_cluster.vdc.endpoint}"
    gcp_region = var.gcp_region
    gcp_zone = var.gcp_zone
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
    "docker_root_image": "gcr.io/${var.gcp_project}/ubuntu:18.04",
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

resource "google_service_account" "gcr_pull" {
  account_id   = "gcr-pull"
  display_name = "pull from gcr.io"
}

resource "google_service_account_key" "gcr_pull_key" {
  service_account_id = google_service_account.gcr_pull.name
}

resource "google_service_account" "gcr_push" {
  account_id   = "gcr-push"
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

resource "google_storage_bucket_iam_member" "gcr_push_admin" {
  bucket = google_container_registry.registry.id
  role = "roles/storage.admin"
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
