output "credentials" {
  value = {
    appId       = var.application_id
    displayName = azuread_service_principal.sp.display_name
    password    = azuread_application_password.password.value
    tenant      = azuread_service_principal.sp.application_tenant_id
    objectId    = azuread_service_principal.sp.object_id
    appObjectId = var.object_id
  }
  sensitive = true
}

output "principal_id" {
  value = azuread_service_principal.sp.object_id
}
