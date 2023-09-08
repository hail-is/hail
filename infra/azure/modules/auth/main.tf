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
module "testns_auth_sp" {
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

resource "random_uuid" "hailctl_oauth2_idenfier_uri_id" {}
resource "random_uuid" "hailctl_oauth2_scope_id" {}

resource "azuread_application" "hailctl_oauth2" {
  display_name = "${var.resource_group_name}-hailctl-oauth2"

  identifier_uris = ["api://hail-${random_uuid.hailctl_oauth2_idenfier_uri_id.result}"]

  public_client {
    redirect_uris = ["http://localhost/"]
  }

  api {
    oauth2_permission_scope {
      admin_consent_description  = "Allow the Hail library to access the Hail Batch service on behalf of the signed-in user."
      admin_consent_display_name = "hailctl"
      user_consent_description   = "Allow the Hail library to access the Hail Batch service on your behalf."
      user_consent_display_name  = "hailctl"
      enabled                    = true
      id                         = random_uuid.hailctl_oauth2_scope_id.result
      type                       = "User"
      value                      = "batch.default"
    }
  }
}

locals {
  oauth2_credentials = {
    appId    = azuread_application.oauth2.application_id
    password = azuread_application_password.oauth2.value
    tenant   = data.azurerm_client_config.primary.tenant_id
  }

  appIdentifierUri = "api://hail-${random_uuid.hailctl_oauth2_idenfier_uri_id.result}"
  userOauthScope   = "${local.appIdentifierUri}/batch.default"
  spOauthScope     = "${local.appIdentifierUri}/.default"

  hailctl_oauth2_credentials = {
    appId            = azuread_application.hailctl_oauth2.application_id
    appIdentifierUri = local.appIdentifierUri
    # For some reason SP client secret authentication refuses scopes that are not .default and this returned a valid token with
    # the desired audience. When creating the oauth scope, terraform refused to create a scope that started with a `.` e.g. `.default`, and
    # as such was forced to create the scope `batch.default`. Whether this is a bug in the terraform provider or a feature of AAD is unclear.
    userOauthScope   = local.userOauthScope
    spOauthScope     = local.spOauthScope
    tenant           = data.azurerm_client_config.primary.tenant_id
  }
}

resource "kubernetes_secret" "auth_oauth2_client_secret" {
  metadata {
    name = "auth-oauth2-client-secret"
  }

  data = {
    "client_secret.json" = jsonencode(local.oauth2_credentials)
    "hailctl_client_secret.json" = jsonencode(local.hailctl_oauth2_credentials)
    "sp_oauth_scope" = local.spOauthScope
  }
}
