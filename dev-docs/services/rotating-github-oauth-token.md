# Rotating the CI GitHub OAuth Token

The `hail-ci-0-1-github-oauth-token` Kubernetes secret stores a GitHub fine-grained
personal access token (PAT) used by CI to interact with GitHub (posting status checks,
commenting on PRs, creating releases, and pushing to DSP repositories).

This token has an expiration date set at creation time and must be rotated before it
expires.

Reminder of the setup: 
- The token is stored in a sops-encrypted JSON file in the repository.
- The JSON file is read in as a configuration for Terraform.
- Terraform updates the `hail-ci-0-1-github-oauth-token` Kubernetes secret with the new token value.
- CI loads the token from the Kubernetes secret in order to interact with GitHub.

## Prerequisites

- A GitHub account with permission to create fine-grained PATs for the Github organization holding the repository being managed by CI
- Access to the sops encryption key (GCP KMS) for the target deployment
- Terraform access for the target infra directory
- `kubectl` access to the cluster

## 1. Create a New Fine-Grained PAT

Go to GitHub **Settings > Developer settings > Personal access tokens > Fine-grained tokens**
and create a new token.

Key settings:
- **Resource owner**: the Github organization holding the repository being managed by CI
- **Expiration**: set an appropriate expiration (note the date so you know when to rotate next)
- **Repository access and permissions**: match the scopes of the existing token

## 2. Update the Sops-Encrypted Config

> [!NOTE]
> The examples below are for the GCP deployment managed in the `gcp-broad` directory. 
> The process is similar for other deployments, but the config file path will be different.

```sh
# Ensure you're authenticated with the correct GCP project so sops can access the KMS key
gcloud auth application-default login

# Edit the encrypted config (sops decrypts in-place for editing, re-encrypts on save)
sops infra/gcp-broad/hail-is/ci_config.enc.json
```

Replace the value of `github_oauth_token` with the new PAT. If `github_user1_oauth_token`
also needs rotation, update it as well.

For other deployments, update the equivalent input:
- **`infra/gcp/`**: same sops flow, with the config file under the appropriate
  `<github_organization>/ci_config.enc.json` path
- **`infra/k8s/`**: update the `ci_and_deploy_github_oauth_token` Terraform variable
  in whatever tfvars or input mechanism is used for that deployment

## 3. Apply Terraform

```sh
cd infra/gcp-broad   # or the appropriate infra directory
terraform apply
```

This updates the `hail-ci-0-1-github-oauth-token` Kubernetes secret with the new token
value.

## 4. Restart the CI Pod

The CI deployment mounts the secret as a volume, so existing pods won't see the update
until they restart.

```sh
kubectl rollout restart deployment/ci
```

Alternatively, wait for the next CI deploy to pick up the new secret.

## 5. Verify

- Check that CI can post status checks on a test PR
- If close to a release, verify the release flow can create GitHub releases (the token
  is used in `build.yaml` for release jobs via `/hail-ci-0-1-github-oauth-token/oauth-token`)

## 6. Revoke the Old Token

Once the new token is confirmed working:

1. Go to GitHub **Settings > Developer settings > Personal access tokens > Fine-grained tokens**
2. Find the old token and delete it

## 7. Commit the Updated Sops File

If you updated a sops-encrypted config file, commit and merge the change:

```sh
git add infra/gcp-broad/hail-is/ci_config.enc.json
git commit -m '[ci] rotate github oauth token'
```

## Where the Token Is Used

For reference, the token flows through the system as follows:

| Layer | Location |
|-------|----------|
| Sops config | `infra/gcp-broad/hail-is/ci_config.enc.json` (field: `github_oauth_token`) |
| Terraform | `infra/gcp-broad/ci/main.tf` → `kubernetes_secret.hail_ci_0_1_github_oauth_token` |
| K8s secret | `hail-ci-0-1-github-oauth-token` (key: `oauth-token`) |
| CI deployment | `ci/deployment.yaml` — mounted at `/secrets/oauth-token`, read via `HAIL_CI_OAUTH_TOKEN` env var |
| Release builds | `build.yaml` — mounted at `/hail-ci-0-1-github-oauth-token`, used for GitHub release creation and `GIT_ASKPASS` |
