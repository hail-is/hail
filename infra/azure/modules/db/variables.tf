variable resource_group {
  type = object({
    name     = string
    location = string
  })
}

variable subnet_id {
  type = string
}
