resource "azuread_service_principal" "sp" {
  application_id = var.application_id
}

resource "azuread_application_password" "password" {
  application_object_id = var.object_id
}
