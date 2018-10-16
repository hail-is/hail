# Secrets
## Testing of PRs

PR CI test pods for `github.com/hail-is/hail` mount a secret named
`hail-ci-0-1-service-account-key`. This secret is expected to contain four files:

 - `hail-ci-0-1.key`, a GCP service account key with sufficient privileges to
   create folders and upload files to `gs://hail-ci-0-1` (i.e. `objectCreator`)

 - `oauth-token`, `user1`, `user2`, see CI's [SECRETS.MD](./ci/SECRETS.md)

## Deploying

CI deployment pods for `github.com/hail-is/hail` mount a secret named
`hail-ci-deploy-hail-is-hail`. This secret is expected to contain two files:

 - `ci-deploy-0-1--hail-is-hail.json`, a GCP service account key with sufficient
   privileges to create sub directories and files in `gs://hail-common`

 - `gcr-push-service-account-key.json`, a GCP service account key with
   sufficient privileges to push to the project's Google Container Registry and
   to issue `kubectl` commands.
