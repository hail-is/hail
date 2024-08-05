# Hail Identity Management

Every application in the Hail System assumes an identity from the
Identity Provider (IdP) of the backing cloud environment.
In GCP, these identities are human Google accounts
or Google Service Accounts. In Azure, they are human users,
Service Principals, and Managed Identities in Microsoft Entra ID
(aka Active Directory).


## Identity Assignment

Which identity is assumed by an application can sometimes depend
on where the application is running. These are separated into the
following environments.


### User's Computer

Code running in this environment acts under the user's human identity. Authorization
to act on the user's behalf is obtained through an OAuth2 flow, either through
auth.hail.is/login or `hailctl auth login`.

NOTE: The credentials that are obtained through `hailctl auth login` are purely
to interact with the Hail Service's APIs and are narrowly scoped.
Requests directly to the cloud use the credentials obtained through the clouds'
CLIs, `gcloud` and `az`. For example, you can use the hail FS by running
`gcloud auth application-default login` or setting `GOOGLE_APPLICATION_CREDENTIALS`
without having an account with the Hail Service.


### User Code in the Cloud

Hail Batch jobs run processes on the user's behalf, and therefore need to assume an
identity that represents the user. Since we cannot (and do not want to) have access
to a user's human credentials or OAuth flow on every job, the Hail System creates and
maintains robot identities to represent the user. In GCP, these are Google Service
Accounts (GSAs) and in Azure they are Service Principals. The Auth service
is responsible for the lifecycle of these robot identities, and the Batch service is
responsible for securing credentials for these identities and delivering them to
Batch Workers. Service operators are responsible for manually rotating these credentials
on a regular cadence using the `devbin/rotate_keys.py` script.


### Hail Services

Services like `auth`, `batch` and `ci` have their own robot identities just like user
robot accounts. Unlike user robot accounts, these identities are granted certain roles
in the cloud environment that allow them to perform their functions like creating VMs
and writing to buckets. These roles should be tightly restricted to just the permissions
needed by the specific service.


## Authentication and Authorization

The Hail System authenticates and authorizes requests in the system through OAuth2
access tokens from the underlying IdP. See [this RFC](https://github.com/hail-is/hail-rfcs/blob/main/rfc/0001-oauth-access-tokens.rst)
for details on how this is implemented.

When an application needs to use a Hail API, it creates an access token either through
a metadata server or on-disk credentials. See [the keyless job RFC](https://github.com/hail-is/hail-rfcs/blob/main/rfc/0012-keyless-job-auth.rst)
for details on how this works for Batch Workers which are multi-tenant. The Auth service
inspects the access token to discover the identity behind the request and validates the
`aud` claim to ensure the token is intended for the Hail API. It is up to the service hosting the
targeted API endpoint to validate that the authenticated user is authorized to use the endpoint.


## Legacy authentication methods

Prior to using access tokens, the Auth service would issue long-lived API keys
that would be persisted on users' computers and mounted in Batch job containers.
At time of writing, these tokens are no longer used in Hail clients but support
has yet to be officially dropped.

In addition to long-lived API keys, the Auth service also supports "Copy-Paste tokens"
which are short-lived, on-demand tokens that users can obtain to then grant temporary
access to another user-controlled computer where they cannot conduct an OAuth flow.
The primary motivating use case here is to access Hail Batch from a Terra Jupyer notebook.
This authentication mechanism was originally implemented as a workaround for environments with
limited identity control and not a long-term solution for authenticating from such environments.
