terraform {
  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "2.2.0"
    }
  }
  # TODO Make this work for google too
  backend "azurerm" {}
}

locals {
  docker_root_image = "${var.global_config.global.docker_prefix}/ubuntu:20.04"
}

provider "kubernetes" {
  config_path = "~/.kube_config"
}

resource "kubernetes_secret" "registry_push_credentials" {
  metadata {
    name = "registry-push-credentials"
  }

  data = {
    "credentials.json" = jsonencode(var.registry_push_credentials)
  }
}

resource "kubernetes_secret" "batch_worker_ssh_public_key" {
  metadata {
    name = "batch-worker-ssh-public-key"
  }

  data = {
    "ssh_rsa.pub" = file("~/.ssh/batch_worker_ssh_rsa.pub")
  }
}

resource "kubernetes_secret" "zulip_config" {
  metadata {
    name = "zulip-config"
  }

  data = {
    ".zuliprc" = file("~/.hail/.zuliprc")
  }
}

resource "kubernetes_secret" "auth_oauth2_client_secret" {
  metadata {
    name = "auth-oauth2-client-secret"
  }

  data = {
    "client_secret.json"    = jsonencode(var.oauth2_credentials)
    "client_secret_v2.json" = jsonencode(var.oauth2_credentials)
  }
}

resource "kubernetes_secret" "gsa_key" {
  for_each = var.service_credentials
  metadata {
    name = "${each.key}-gsa-key"
  }

  data = {
    "key.json" = jsonencode(each.value)
  }
}
