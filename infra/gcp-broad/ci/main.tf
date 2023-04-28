resource "random_string" "hail_ci_bucket_suffix" {
  length = 5
}

resource "google_storage_bucket" "bucket" {
  name = "hail-ci-${random_string.hail_ci_bucket_suffix.result}"
  location = var.bucket_location
  force_destroy = false
  storage_class = var.bucket_storage_class
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
  role = "roles/storage.legacyBucketWriter"
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

resource "kubernetes_secret" "zulip_config" {
  count = fileexists("~/.hail/.zuliprc") ? 1 : 0
  metadata {
    name = "zulip-config"
  }

  data = {
    ".zuliprc" = fileexists("~/.hail/.zuliprc") ? file("~/.hail/.zuliprc") : ""
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
