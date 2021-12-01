variable "global_config" {
  type = object({
    global = object({
      cloud = string,
      domain = string,
      internal_ip = string,
      ip = string,
      kubernetes_server = string,
      docker_prefix = string,
      batch_logs_storage_uri = string,
      test_storage_uri = string,
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
    instance = string,
    connection_name = string,
  })
}

variable "registry_push_credentials" {
  type = map
}

variable "service_credentials" {
  type = map
}
