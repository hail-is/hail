variable resource_group {
  type = object({
    name     = string
    location = string
  })
}

variable k8s_default_node_pool_machine_type {
  type = string
}

variable k8s_user_pool_machine_type {
  type = string
}

variable k8s_preemptible_node_pool_name {
  type = string
}

variable k8s_nonpreemptible_node_pool_name {
  type = string
}

variable container_registry_id {
  type = string
}
