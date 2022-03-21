variable resource_group {
  type = object({
    name     = string
    location = string
  })
}

variable vnet_id {
  type = string
}

variable subnet_id {
  type = string
}
