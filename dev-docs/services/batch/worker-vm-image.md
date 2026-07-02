# Batch Worker VM Image

## Background

Batch worker VMs use a two-layer image system:

1. **VM image** (this doc) -- A GCE VM image with the base OS, Docker, GPU drivers, and the root
   Docker image pre-pulled. This is what each worker VM boots from.

2. **Docker worker image** (`Dockerfile.worker`) -- The container that actually runs the batch
   worker process. It's built separately via `make batch-worker-image` and pulled onto workers at
   startup.

The VM image exists primarily so that worker VMs boot fast. Without it, every new worker would
need to install Docker, pull a ~2GB root image, install GPU drivers, etc. on every boot. By baking
all of that into a VM image, new workers go from creation to running jobs in under a minute.

The VM image changes rarely (only when we need to update Docker, GPU drivers, or other
OS-level dependencies). The Docker worker image changes on every deploy.

### How the build works

The build scripts create a temporary VM from a stock Ubuntu image, run a startup script that
installs everything, wait for the VM to shut itself down, then snapshot its disk into a reusable
image. The temporary VM is deleted afterward.

## GCP

### First time creation

If this is a brand new deployment, the worker image must be created before Batch can be deployed.
See `infra/gcp/README.md` — the image creation step comes after `deploy_unmanaged` and before
downloading global-config.

#### Prerequisites

- `gcloud` configured and authenticated with the target project
- `NAMESPACE` environment variable set (usually `default`)
- Access to the project's global-config (for project ID, zone, docker root image)

#### Building the image

From the repo root:

```bash
NAMESPACE=default batch/gcp-create-worker-image.sh
```

The script will show a confirmation prompt with the image name, version, project, and zone before
proceeding.

#### What gets installed (startup script)

The GCP startup script (`build-batch-worker-image-startup-gcp.sh`) installs:

- Google logging agent and Cloud Ops agent
- Docker CE + `docker-credential-gcr`
- NVIDIA drivers (535.183.01) and `nvidia-container-toolkit` — baked into every worker image, but
  only activated at runtime on VMs with GPUs attached (see `is_gpu()` check in `worker.py`)
- Build tools (gcc-12, g++-12)
- Pre-pulls the `docker_root_image` from Artifact Registry

The VM shuts itself down when the script completes. The build script polls for this, then snapshots
the disk.

#### Image naming

- `default` namespace: `batch-worker-{VERSION}` (e.g. `batch-worker-17`)
- Other namespaces: `batch-worker-{NAMESPACE}-{VERSION}`

### Incrementing the version

#### At the same time

1. Take this opportunity to identify a suitable new base image (`UBUNTU_IMAGE` - look in the image list in GCP console for the latest).
2. Also check whether there are new nvidia GPU drivers (I've seen out of date drivers cause the build to hang indefinitely if they don't
   match the OS version)

#### Procedure to Increment

1. Increment `WORKER_IMAGE_VERSION` in `batch/gcp-create-worker-image.sh`
    - Very Important! You MUST increment the version before running the script! The version is part of
the image name, so running without doing this would replace the current image relied on in prod.
2. Run the build script with a custom NAMESPACE, to make sure the image builds and deploys successfully.
    - eg: `NAMESPACE=YOURNAME batch/gcp-create-worker-image.sh`
3. Monitor the build process: open the VM list in gcloud console. find the build worker you just triggered. Under the three dots
open Monitoring and watch the logs.
4. Update the hardcoded image reference in
   `batch/batch/cloud/gcp/driver/create_instance.py` (search for `batch-worker-`)
5. Test deploy, and check things look good.
6. Repeat with `NAMESPACE=default`
7. Create PR, watch CI, deploy with the updated image.
8. NOTE: The rollout will only impact newly created workers. Any preexisting workers will remain on their
old image until they get manually deleted, or gradually replaced through normal system operation.

#### Rollback

Rollback is fortunately easy: simply revert the version in `batch/batch/cloud/gcp/driver/create_instance.py` and
redeploy. You'll need to delete all existing workers for the rollback to take effect. 

## Key files

| File | Purpose |
|------|---------|
| `batch/gcp-create-worker-image.sh` | GCP build orchestration script |
| `batch/build-batch-worker-image-startup-gcp.sh` | GCP VM startup/provisioning (Jinja2 template) |
| `batch/Dockerfile.worker` | Docker worker image (separate from the VM image) |
| `batch/batch/cloud/gcp/driver/create_instance.py` | Runtime: creates worker VMs using the GCP image |
