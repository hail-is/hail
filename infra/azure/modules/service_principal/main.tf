resource "azuread_service_principal" "sp" {
  application_id = var.application_id
}

resource "azuread_application_password" "password" {
  application_object_id = var.application_object_id
}

locals {
  credentials = {
    appId       = var.application_id
    displayName = azuread_service_principal.sp.display_name
    password    = azuread_application_password.password.value
    tenant      = azuread_service_principal.sp.application_tenant_id
    objectId    = azuread_service_principal.sp.object_id
    appObjectId = var.application_object_id
  }
}

resource "kubernetes_secret" "gsa_key" {
  metadata {
    name = "${var.name}-gsa-key"
  }

  data = {
    "key.json" = jsonencode(local.credentials)
  }
}

resource "azurerm_role_assignment" "subscription_roles" {
  for_each = toset(var.subscription_roles)

  scope                = var.subscription_resource_id
  role_definition_name = each.key
  principal_id         = azuread_service_principal.sp.object_id
}

resource "azurerm_role_assignment" "resource_group_roles" {
  for_each = toset(var.resource_group_roles)

  scope                = var.resource_group_id
  role_definition_name = each.key
  principal_id         = azuread_service_principal.sp.object_id
}
