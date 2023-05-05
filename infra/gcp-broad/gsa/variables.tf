variable "name" {
  type = string
}

variable project {
  type = string
}

variable "iam_roles" {
  type = list(string)
  default = []
}
