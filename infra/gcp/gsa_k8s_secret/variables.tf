variable "name" {
  type = string
}

variable "iam_roles" {
  type = list(string)
  default = []
}
