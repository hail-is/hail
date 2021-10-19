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
  docker_root_image = "${var.global_config.global.docker_prefix}/ubuntu:18.04"
}

provider "kubernetes" {
  config_path = "~/.kube_config"
}

resource "kubernetes_secret" "acr_push_credentials" {
  metadata {
    name = "acr-push-credentials"
  }

  data = {
    "credentials.json" = jsonencode(var.acr_push_credentials)
  }
}
resource "kubernetes_secret" "zulip_config" {
  metadata {
    name = "zulip-config"
  }

  data = {
    ".zuliprc" = file("~/.zuliprc")
  }
}

resource "kubernetes_secret" "auth_oauth2_client_secret" {
  metadata {
    name = "auth-oauth2-client-secret"
  }

  data = {
    "client_secret.json" = file("~/auth_oauth2_client_secret.json")
  }
}
