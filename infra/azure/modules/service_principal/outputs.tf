output principal_id {
  value = azuread_service_principal.sp.object_id
}

output credentials {
  value = local.credentials
  sensitive = true
}
