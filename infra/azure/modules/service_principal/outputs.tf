output display_name {
  value = azuread_service_principal.sp.display_name
}

output application_id {
  value = azuread_service_principal.sp.application_id
}

output principal_id {
  value = azuread_service_principal.sp.object_id
}

output credentials {
  value = local.credentials
  sensitive = true
}
