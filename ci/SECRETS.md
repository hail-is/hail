# Secrets
## Runtime

The CI needs two secrets in the working directory:

 - `./oauth-token/oauth-token`, a GitHub token with `repo:status` and
   `public_repo` used to merge PRs and update statuses

 - `./gcloud-token/hail-ci-0-1.key`, a GCP service account key with
   `objectCreator` for `gs://hail-ci-0-1`

## Test

For testing, the CI needs access tokens in `./github-tokens`, see the
[README.md](./github-tokens/README.md), and also the two secrets needed for
runtime.
