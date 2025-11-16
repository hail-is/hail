# Updating third party images

This process updates third party images in cloud artifact registries after changes in [docker/third-party/images.txt](../../docker/third-party/images.txt) .

>[!NOTE]
> As of 2025-10-07 this process is run automatically in CI deploy builds but not during CI runs, so you will need to manually run the following steps to update the images for PRs using the new images to succeed.

## Prerequisites

### Install skopeo

Not absolutely necessary but installing skopeo will make this go a lot smoother - and won't run into docker
"no space left on device" issues.

```bash
$ brew install skopeo
```

### Set the `kubectl` Context

Why do we need kubectl? Because this allows config.mk to access the global-config secret's .data.docker_prefix field. 

See [Setting the `kubectl` Context](setting_the_kubectl_context.md).

### Connect or fetch credentials to the container registry

Why? Because we need to push the third party images into the container registry.

See [Connecting Docker to Container Registry Credentials](connecting_docker_to_container_registry_creds.md).

 
## Run the following command to update the image:

```bash
$ NAMESPACE=default make -C  docker/third-party copy
```
