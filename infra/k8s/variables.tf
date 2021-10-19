variable "global_config" {
  type = object({
    global = object({
      cloud = string,
      domain = string,
      internal_ip = string,
      ip = string,
      kubernetes_server = string,
      docker_prefix = string,
    }),
    azure = map(any)
  })
}

variable "sql_config" {
  type = object({
    server_ca_cert = string,
    client_cert = string,
    client_private_key = string,
    host = string,
    user = string,
    password = string,
  })
}

variable "acr_push_credentials" {
  type = map
}
