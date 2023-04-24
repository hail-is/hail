variable "github_oauth_token" {
  type = string
  sensitive = true
}

variable "github_user1_oauth_token" {
  type = string
  sensitive = true
}

variable "bucket_location" {
  type = string
}

variable "bucket_storage_class" {
  type = string
}

variable "watched_branches" {
  type = list(tuple([string, bool, bool]))
}

variable "deploy_steps" {
  type = list(string)
}

variable "ci_email" {
  type = string
}

variable "container_registry_id" {
  type = string
}

variable "github_context" {
  type = string
}
