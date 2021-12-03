variable "resource_group" {
  type = object({
    name     = string
    location = string
  })
}

variable "ci_principal_id" {
  type = string
}

variable "container_registry_id" {
  type = string
}
