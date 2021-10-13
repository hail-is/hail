terraform {
  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "2.2.0"
    }
  }
  # backend "azurerm" {}
}

locals {
  docker_root_image = "${var.global_config.global.docker_prefix}/ubuntu:18.04"
}

provider "kubernetes" {
  config_path = "~/.kube/config"
}
