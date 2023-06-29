variable "resource_group" {
  type = object({
    name     = string
    location = string
  })
}

variable "ci_principal_id" {
  type = string
}

variable "container_registry_id" {
  type = string
}

variable "test_storage_container_resource_id" {
  type = string
}

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

variable "github_context" {
  type = string
}

variable "storage_account_suffix" {
  type = string
}
