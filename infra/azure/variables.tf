variable az_resource_group_name {
  type = string
}

variable domain {
  type = string
}

variable acr_name {
  type    = string
  default = ""
}

variable acr_sku {
  type    = string
  default = "Premium"
}

variable k8s_default_node_pool_machine_type {
  type    = string
  default = "Standard_D2_v2"  # 2 vCPU
}

variable k8s_user_pool_machine_type {
  type    = string
  default = "Standard_D3_v2"  # 4 vCPU
}

variable k8s_preemptible_node_pool_name {
  type    = string
  default = "preempt1"
}

variable k8s_nonpreemptible_node_pool_name {
  type    = string
  default = "nonpreempt1"
}

variable organization_domain {
  type = string
}

variable batch_test_user_storage_account_name {
  type = string
}

variable "ci_config" {
  type = object({
    ci_and_deploy_github_oauth_token = string
    ci_test_repo_creator_github_oauth_token = string
    watched_branches = list(tuple([string, bool, bool]))
    deploy_steps = list(string)
    github_context = string
  })
  default = null
}

variable oauth2_developer_redirect_uris {
  type    = list(string)
  default = []
}

variable storage_account_suffix {
  type = string
}
