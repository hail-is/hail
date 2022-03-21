variable name {
  type = string
}

variable application_id {
  type = string
}

variable application_object_id {
  type = string
}

variable subscription_resource_id {
  type = string
  default = ""
}

variable subscription_roles {
  type = list(string)
  default = []
}

variable resource_group_id {
  type = string
  default = ""
}

variable resource_group_roles {
  type = list(string)
  default = []
}
