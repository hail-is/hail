# Docker Hub Cache/Proxy

Hail uses a Google Artifact Registry (GAR) remote repository as a proxy/cache for Docker Hub images to mitigate rate limiting issues when pulling images from Docker Hub.

## Overview

When worker VMs or CI/CD processes pull Docker images, they can encounter Docker Hub rate limits, especially when multiple VMs share the same outbound IP (e.g., behind CloudNAT). To address this, we use a GAR remote repository that acts as a pull-through cache for Docker Hub.

### How It Works

1. **GAR Remote Repository**: A Google Artifact Registry repository configured as a remote repository targeting Docker Hub.
2. **Image Rewriting**: Docker Hub image references are automatically rewritten to use the GAR proxy before pulling.
3. **Transparent Caching**: The GAR repository caches images on first pull, and subsequent pulls use the cached version, reducing requests to Docker Hub.

## Configuration

### Terraform Setup

The Docker Hub proxy repo is created in Terraform (`infra/gcp/main.tf` and `infra/gcp-broad/main.tf`) as an Artifact Registry repository with the mode set to `REMOTE` and the target set to `DOCKER_HUB`.

Note: By default the repository is configured to pull anonymously from Docker Hub. That setup is very likely to run into the toomanyrequests error from dockerhub because AR, as a global service, reuses the same IP address for multiple requests across multiple users and projects. To avoid this, it is recommended to create a "hail system" docker hub account, create a personal access token, and configure the repository to pull from Docker Hub using that token manually in the google cloud console.

### IAM Permissions

The following IAM bindings grant access:

1. **Batch Agent**: The `batch2-agent` service account has `roles/artifactregistry.reader` on the `dockerhubproxy` repository.
2. **Pet Service Accounts**: A Google Cloud Identity group (`pet_service_accounts@{organization_domain}`) has `roles/artifactregistry.reader` on the repository. User shadow service accounts are automatically added to this group when created.

### Global Config

The `dockerhub_prefix` is stored in the `global-config` Kubernetes secret and made available to:

- Batch services via environment variables (`DOCKERHUB_PREFIX`)
- CI/CD processes via build configuration
- Makefile targets via the `DOCKERHUB_PREFIX` variable from config.mk

## Image Rewriting

In general, image names are still specified in code in their natural form. Before pulling an image therefore, we should update or rewrite the image to make sure it is pulled from the proxy registry (if configured).

Image names can be specified as:

- Bare images (e.g., `ubuntu:24.04`)
- Organization images (e.g., `envoyproxy/envoy:v1.33.0`)
- Registry images (e.g., `us-central1-docker.pkg.dev/my-project/my-repo/my-image:latest`)

The rewriting rules are:

- Bare images are rewritten to `{dockerhub_prefix}/library/{image_name}` (eg `us-central1-docker.pkg.dev/my-project/dockerhubproxy/library/ubuntu:24.04`)
- Organization images are rewritten to `{dockerhub_prefix}/{organization}/{image_name}` (eg `us-central1-docker.pkg.dev/my-project/dockerhubproxy/envoyproxy/envoy:v1.33.0`)
- Registry images are not rewritten (eg `us-central1-docker.pkg.dev/my-project/my-repo/my-image:latest`)

### Job Submission

At job submission time, if the `dockerhub_proxy` feature flag is enabled and the global-config `dockerhub_prefix` variable is set, the image name is rewritten to use the `DOCKERHUB_PREFIX` variable.

### CI/CD Processes

Within CI/CD builds, the `DOCKERHUB_PREFIX` variable is automatically set to the `dockerhub_prefix` value from the `global-config` secret.

Each task which interacts with Docker Hub images should apply the rewriting rules to the image name before pulling the image.

- Some examples of CI/CD build tasks which have been updated to use `DOCKERHUB_PREFIX` are:
  - `copy_third_party_images` in `build.yaml`
  - `mirror_hailgenetics_images` in `build.yaml`


### Makefile Targets

Makefile targets use config.mk to set the `DOCKERHUB_PREFIX` variable. Thereafter if they need to interact with Docker Hub images, they should apply the rewriting rules to the image name before pulling the image. 


Note: It's likely that underlying scripts are shared between the CI/CD tasks and the Makefile targets, so the rewriting rules should be applied in the shared script. Then the only responsibility of the Makefile targets is to set the `DOCKERHUB_PREFIX` variable.

Some examples of Makefile targets which have been updated to use `DOCKERHUB_PREFIX`:
  - `docker/third-party/Makefile` target: `copy`
  - `docker/hailgenetics/Makefile` target: `mirror-dockerhub-images`

## Usage

### System

So long as the `DOCKERHUB_PREFIX` variable is set, the rewriting rules will be applied to the image name before pulling the image in all of:
- Batch services (at job submission time)
- CI/CD builds
- Makefile targets

### Manual Image Pulls

If you need to manually pull an image through the proxy, you can construct the path:

```bash
# Example: Pulling ubuntu:24.04 through the proxy
REGION=us-central1
PROJECT=your-project
IMAGE=ubuntu:24.04
docker pull ${REGION}-docker.pkg.dev/${PROJECT}/dockerhubproxy/library/${IMAGE}
```

Note that from your local machine you will need to make sure you have logged in to gcloud as a user who is allowed to access the `dockerhubproxy` repository. You must also run `gcloud auth configure-docker` to configure docker to use your gcloud credentials when making docker pull calls.
