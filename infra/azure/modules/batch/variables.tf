variable resource_group {
  type = object({
    id       = string
    name     = string
    location = string
  })
}

variable batch_test_user_storage_account_name {
  type = string
}

variable container_registry_id {
  type = string
}

variable storage_account_suffix {
  type = string
}
