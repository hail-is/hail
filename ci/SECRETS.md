# Secrets
## Runtime

The CI needs two secrets in the working directory:

 - `./oauth-token/oauth-token`, a GitHub personal access token with
   `repo:status` and `public_repo` used to merge PRs and update statuses

 - `./gcloud-token/hail-ci-0-1.key`, a GCP service account key with
   `objectCreator` for `gs://hail-ci-0-1`

## Test

The CI tests need four secrets:

 - `/secrets/oauth-token`, a GitHub personal access token with the same privileges as
   the runtime's `oauth-token`

 - `/secrets/hail-ci-0-1.key`, a GCP service account key with the same
   privileges as the runtime's `hail-ci-0-1.key`

 - `/secrets/user1` and `/secrets/user2`, two distinct GitHub personal access
   tokens which will be used for integration testing with GitHub. The tokens
   need the following privileges:
   - `admin:repo_hook`, used to set up CI webhooks for test repos
   - `delete_repo`, used to delete the repo after tests are finished,
   - `repo`, used to commit, update status, and review PRs on a repo

