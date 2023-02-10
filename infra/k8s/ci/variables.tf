variable "ci_and_deploy_github_oauth_token" {
  type = string
}

variable "ci_test_repo_creator_github_oauth_token" {
  type = string
}

variable "watched_branches" {
  type = list(tuple([string, bool, bool]))
}

variable "deploy_steps" {
  type = list(string)
}

variable "test_oauth2_callback_urls" {
  type = list(string)
}

variable "storage_uri" {
  type = string
}

variable "github_context" {
  type = string
}
