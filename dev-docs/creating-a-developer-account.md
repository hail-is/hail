# Creating a Developer Account

Do *not* sign up for a developer account. The "sign up" link on "auth.hail.is"
can only produce non-developer accounts. Instead, an extant developer must:

- Navigate to https://auth.hail.is/users and create the new user. Use their
  Broad Institute email and username. Make sure you check the "Developer"
  checkbox and not the "Service Account" checkbox.
- Unfortunately it is impossible to automatically register an OAuth 2.0 redirect
  URI for the new namespace. Instead:
  - Navigate to https://console.developers.google.com/apis/credentials/?project=broad-ctsa
    (nb: the `broad-ctsa` project, not `hail-vdc`).
  - Click "auth" under "OAuth 2.0 Client IDs".
  - Add `https://internal.hail.is/${USERNAME}/auth/oauth2callback` to the list
    of "Authorized redirect URIs".


# Programmatic manipulation of OAuth 2.0 Client IDs

There is a [GitHub
issue](https://github.com/hashicorp/terraform-provider-google/issues/6074)
explaining that Google does not provide a public API to manipulate OAuth 2.0
Client ID redirect URIs, much to everyone's chagrin.

Google marked as fixed [an issue to create an
API](https://issuetracker.google.com/issues/116182848) for modifying OAuth 2.0
Client IDs even though all they did was provide a very limited API:
- https://cloud.google.com/iap/docs/reference/rest#rest-resource:-v1.projects.brands
- https://cloud.google.com/iap/docs/programmatic-oauth-clients

The `gcloud alpha iap oauth-clients list` command does not list our OAuth 2.0
Client ID. Presumably the type of client id that supports redirect URIs is
special and completely unsupported by this API.
