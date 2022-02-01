variable resource_group {
  type = object({
    id       = string
    name     = string
    location = string
  })
}

variable container_registry_id {
  type = string
}
