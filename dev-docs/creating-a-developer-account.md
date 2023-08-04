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

# Google IAM and Kubernetes Roles

In order to access the Kubernetes cluster, the extant developer should do the
following:

- Navigate in the GCP Console to IAM and grant the IAM `Kubernetes Engine Cluster Viewer` role.
  This will allow the new developer's google account to authenticate with the
  cluster using `kubectl` but it will not grant the new developer access to k8s resources.
- To grant the new developer access to their developer namespace, the extant
  developer should apply the following configuration through `kubectl apply`.

```yaml
kind: Role
apiVersion: rbac.authorization.k8s.io/v1
metadata:
  name: dev-admin
  namespace: <DEV_USERNAME>
rules:
- apiGroups: [""]
  resources: ["*"]
  verbs: ["*"]
- apiGroups: ["apps"]
  resources: ["*"]
  verbs: ["*"]
- apiGroups: ["rbac.authorization.k8s.io"]
  resources: ["*"]
  verbs: ["*"]
---
kind: RoleBinding
apiVersion: rbac.authorization.k8s.io/v1
metadata:
  name: <DEV_USERNAME>-dev-admin-binding
  namespace: <DEV_USERNAME>
subjects:
- kind: User
  name: <DEV_EMAIL>
  namespace: <DEV_USERNAME>
roleRef:
  kind: Role
  name: dev-admin
  apiGroup: ""
```
