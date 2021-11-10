resource "kubernetes_secret" "ci_config" {
  metadata {
    name = "ci-config"
  }

  data = {
    storage_uri      = var.storage_uri
    deploy_steps     = jsonencode(var.deploy_steps)
    watched_branches = jsonencode(var.watched_branches)
    github_context   = var.github_context
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
