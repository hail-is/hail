variable az_resource_group_name {
  type = string
}

variable subscription_id {
  type = string
}

variable domain {
  type = string
}

variable acr_name {
  type = string
  default = ""
}

variable acr_sku {
  type = string
  default = "Basic"
}

variable k8s_machine_type {
  type = string
  default = "Standard_D2_v2"
}

variable "ci_config" {
  type = object({
    github_oauth_token = string
    github_user1_oauth_token = string
    watched_branches = list(tuple([string, bool, bool]))
    deploy_steps = list(string)
    github_context = string
  })
  default = null
}
