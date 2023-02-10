module "bucket" {
  source        = "../gcs_bucket"
  short_name    = "hail-ci"
  location      = var.bucket_location
  storage_class = var.bucket_storage_class
}

resource "google_storage_bucket_iam_member" "ci_bucket_admin" {
  bucket = module.bucket.name
  role = "roles/storage.objectAdmin"
  member = "serviceAccount:${var.ci_email}"
}

resource "google_storage_bucket_iam_member" "ci_gcr_admin" {
  bucket = var.container_registry_id
  role = "roles/storage.admin"
  member = "serviceAccount:${var.ci_email}"
}

resource "kubernetes_secret" "ci_config" {
  metadata {
    name = "ci-config"
  }

  data = {
    storage_uri               = "gs://${module.bucket.name}"
    deploy_steps              = jsonencode(var.deploy_steps)
    watched_branches          = jsonencode(var.watched_branches)
    test_oauth2_callback_urls = jsonencode(var.test_oauth2_callback_urls)
    github_context            = var.github_context
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
