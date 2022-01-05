variable resource_group {
  type = object({
    name     = string
    location = string
  })
}

variable k8s_machine_type {
  type = string
}

variable container_registry_id {
  type = string
}
