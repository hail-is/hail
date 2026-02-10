# Batch Worker VM Image

## Background

Batch worker VMs use a two-layer image system:

1. **VM image** (this doc) -- A cloud-provider VM image (GCE image / Azure Shared Image Gallery)
   with the base OS, Docker, GPU drivers, and the root Docker image pre-pulled. This is what each
   worker VM boots from.

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

### Prerequisites

- `gcloud` configured and authenticated with the target project
- `NAMESPACE` environment variable set (usually `default`)
- Access to the project's global-config (for project ID, zone, docker root image)

### Building the image

From the repo root:

```bash
NAMESPACE=default batch/gcp-create-worker-image.sh
```

The script will show a confirmation prompt with the image name, version, project, and zone before
proceeding.

### What gets installed (startup script)

The GCP startup script (`build-batch-worker-image-startup-gcp.sh`) installs:

- Google logging agent and Cloud Ops agent
- Docker CE + `docker-credential-gcr`
- NVIDIA drivers (535.183.01) and `nvidia-container-toolkit`
- Build tools (gcc-12, g++-12)
- Pre-pulls the `docker_root_image` from Artifact Registry

The VM shuts itself down when the script completes. The build script polls for this, then snapshots
the disk.

### Image naming

- `default` namespace: `batch-worker-{VERSION}` (e.g. `batch-worker-17`)
- Other namespaces: `batch-worker-{NAMESPACE}-{VERSION}`

### Bumping the version

1. Increment `WORKER_IMAGE_VERSION` in `batch/gcp-create-worker-image.sh`
2. Run the build script as above
3. Update the hardcoded image reference in
   `batch/batch/cloud/gcp/driver/create_instance.py` (search for `batch-worker-`)
4. Deploy batch

### First-time setup

If this is a brand new deployment, the worker image must be created before Batch can be deployed.
See `infra/gcp/README.md` -- the image creation step comes after `deploy_unmanaged` and before
downloading global-config:

```bash
NAMESPACE=default $HAIL/batch/gcp-create-worker-image.sh
```

## Azure

### Prerequisites

- Azure CLI authenticated
- `$HAIL` environment variable pointing to the repo root
- Access to global-config secret (for subscription ID, resource group, location, docker prefix)
- The `batch-worker` managed identity must exist in the resource group

### Building the image

```bash
source $HAIL/devbin/functions.sh
$HAIL/batch/az-create-worker-image.sh
```

### What gets installed (startup script)

The Azure startup script (`build-batch-worker-image-startup-azure.sh`) installs:

- Docker CE
- Azure CLI
- Authenticates with Azure Container Registry via managed identity
- Pre-pulls the `docker_root_image`

Note: Azure workers do not have GPU support baked into the VM image (unlike GCP).

### Image naming

Images are stored in an Azure Shared Image Gallery:

- Gallery: `{RESOURCE_GROUP}_batch`
- Image definition: `batch-worker-22-04`
- Version: `0.0.{N}` (e.g. `0.0.14`)

### Bumping the version

1. Increment `WORKER_VERSION` in `batch/az-create-worker-image.sh`
2. Run the build script as above
3. Update the hardcoded image reference in
   `batch/batch/cloud/azure/driver/create_instance.py` (search for `batch-worker-22-04/versions/`)
4. Deploy batch

## Key files

| File | Purpose |
|------|---------|
| `batch/gcp-create-worker-image.sh` | GCP build orchestration script |
| `batch/az-create-worker-image.sh` | Azure build orchestration script |
| `batch/build-batch-worker-image-startup-gcp.sh` | GCP VM startup/provisioning (Jinja2 template) |
| `batch/build-batch-worker-image-startup-azure.sh` | Azure VM startup/provisioning (Jinja2 template) |
| `batch/Dockerfile.worker` | Docker worker image (separate from the VM image) |
| `batch/batch/cloud/gcp/driver/create_instance.py` | Runtime: creates worker VMs using the GCP image |
| `batch/batch/cloud/azure/driver/create_instance.py` | Runtime: creates worker VMs using the Azure image |
