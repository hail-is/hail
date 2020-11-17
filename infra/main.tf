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

# FIXME rename this to gateway
resource "google_compute_address" "gateway" {
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
    "config.json" = <<END
{
  "gcp_project": "${var.gcp_project}",
  "domain": "${var.domain}",
  "internal_ip": "${google_compute_address.internal_gateway.address}",
  "ip": "${google_compute_address.gateway.address}",
  "kubernetes_server_url": "https://${google_container_cluster.vdc.endpoint}",
  "region": "${var.gcp_region}",
  "zone": "${var.gcp_zone}"
}
END
  }
}
