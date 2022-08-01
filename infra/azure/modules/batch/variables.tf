variable resource_group {
  type = object({
    id       = string
    name     = string
    location = string
  })
}

variable az_storage_account_name {
  type = string
}

variable container_registry_id {
  type = string
}
