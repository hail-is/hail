variable "github_oauth_token" {
  type = string
}

variable "github_user1_oauth_token" {
  type = string
}

variable "watched_branches" {
  type = list(tuple([string, bool, bool]))
}

variable "deploy_steps" {
  type = list(string)
}

variable "storage_uri" {
  type = string
}

variable "github_context" {
  type = string
}
