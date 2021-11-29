variable "resource_group" {
  type = object({
    name     = string
    location = string
  })
}

variable "ci_principal_id" {
  type = string
}
