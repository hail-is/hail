# Auth gateway
A NodeJS app that verifies oauth2 access tokens issued by Auth0, and adds the user id, scope to the response header.

Expected to be used by internal/private services, or by the nginx gateway (as the proxy listed in `auth_request`) that controls access to these private services.

## Environmental Variables
Environmental variables listed in '.env' will be used

Public environmental variables used by the program are listed in env-public

`cat env-public > .env`

### Required environmental variables
AUTH0_DOMAIN
AUTH0_AUDIENCE

Description: https://auth0.com/docs/api-auth/tutorials/verify-access-token
