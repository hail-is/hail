resource "kubernetes_secret" "ci_config" {
  metadata {
    name = "ci-config"
  }

  data = {
    storage_uri               = var.storage_uri
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

resource "kubernetes_secret" "ci_and_deploy_github_oauth_token" {
  metadata {
    name = "hail-ci-0-1-github-oauth-token"
  }

  data = {
    "oauth-token" = var.ci_and_deploy_github_oauth_token
  }
}

resource "kubernetes_secret" "ci_test_repo_creator_github_oauth_token" {
  metadata {
    name = "hail-ci-0-1-service-account-key"
  }

  data = {
    "user1" = var.ci_test_repo_creator_github_oauth_token
  }
}

# Necessary for batch to access secrets/SAs in test namespaces
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
