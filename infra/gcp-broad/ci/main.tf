terraform {
  required_providers {
    sops = {
      source = "carlpett/sops"
      version = "0.6.3"
    }
  }
}

resource "random_string" "hail_ci_bucket_suffix" {
  length = 5
}

resource "google_storage_bucket" "bucket" {
  name = "hail-ci-${random_string.hail_ci_bucket_suffix.result}"
  location = var.bucket_location
  force_destroy = false
  storage_class = var.bucket_storage_class
  uniform_bucket_level_access = true
  labels = {
    "name" = "hail-ci-${random_string.hail_ci_bucket_suffix.result}"
  }
  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      age                        = 7
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

resource "google_storage_bucket_iam_member" "ci_bucket_admin" {
  bucket = google_storage_bucket.bucket.name
  role = "roles/storage.objectAdmin"
  member = "serviceAccount:${var.ci_email}"
}

resource "kubernetes_secret" "ci_config" {
  metadata {
    name = "ci-config"
  }

  data = {
    storage_uri      = "gs://${google_storage_bucket.bucket.name}"
    deploy_steps     = jsonencode(var.deploy_steps)
    watched_branches = jsonencode(var.watched_branches)
    github_context   = var.github_context
    test_oauth2_callback_urls = var.test_oauth2_callback_urls
  }
}

data "sops_file" "zuliprc" {
  count = fileexists("${var.github_organization}/zuliprc.enc") ? 1 : 0
  source_file = "${var.github_organization}/zuliprc.enc"
  input_type = "raw"
}

locals {
    zuliprc = length(data.sops_file.zuliprc) == 1 ? data.sops_file.zuliprc[0] : null
}

resource "kubernetes_secret" "zulip_config" {
  count = local.zuliprc != null ? 1 : 0
  metadata {
    name = "zulip-config"
  }

  data = {
    ".zuliprc" = local.zuliprc.raw
  }
}

resource "kubernetes_secret" "hail_ci_0_1_github_oauth_token" {
  metadata {
    name = "hail-ci-0-1-github-oauth-token"
  }

  data = {
    "oauth-token" = var.github_oauth_token
  }
}

resource "kubernetes_secret" "hail_ci_0_1_service_account_key" {
  metadata {
    name = "hail-ci-0-1-service-account-key"
  }

  data = {
    "user1" = var.github_user1_oauth_token
  }
}
