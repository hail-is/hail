data "azurerm_client_config" "primary" {}

resource "azuread_application" "auth" {
  display_name = "${var.resource_group_name}-auth"

  required_resource_access {
    resource_app_id = "00000003-0000-0000-c000-000000000000"

    resource_access {
      # Application.ReadWrite.All
      id   = "1bfefb4e-e0b5-418b-a88f-73c46d2cc8e9"
      type = "Role"
    }
  }
}
module "auth_sp" {
  source = "../service_principal"

  name                  = "auth"
  application_id        = azuread_application.auth.application_id
  application_object_id = azuread_application.auth.object_id
}

resource "azuread_application" "testns_auth" {
  display_name = "${var.resource_group_name}-testns-auth"

  required_resource_access {
    resource_app_id = "00000003-0000-0000-c000-000000000000"

    resource_access {
      # Application.ReadWrite.All
      id   = "1bfefb4e-e0b5-418b-a88f-73c46d2cc8e9"
      type = "Role"
    }
  }
}
module "test_auth_sp" {
  source = "../service_principal"

  name                  = "testns-auth"
  application_id        = azuread_application.testns_auth.application_id
  application_object_id = azuread_application.testns_auth.object_id
}

resource "azuread_application" "oauth2" {
  display_name = "${var.resource_group_name}-oauth2"

  web {
    redirect_uris = var.oauth2_redirect_uris

    implicit_grant {
      access_token_issuance_enabled = true
      id_token_issuance_enabled     = true
    }
  }
}
resource "azuread_application_password" "oauth2" {
  application_object_id = azuread_application.oauth2.object_id
}

locals {
  oauth2_credentials = {
    appId    = azuread_application.oauth2.application_id
    password = azuread_application_password.oauth2.value
    tenant   = data.azurerm_client_config.primary.tenant_id
  }
}

resource "kubernetes_secret" "auth_oauth2_client_secret" {
  metadata {
    name = "auth-oauth2-client-secret"
  }

  data = {
    "client_secret.json" = jsonencode(local.oauth2_credentials)
  }
}
