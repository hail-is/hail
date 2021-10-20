# Hail on Azure

This is a work in progress for setting up hail infrasture on Azure. The
following should be executed in the `$HAIL/infra/azure` directory unless
otherwise noted.

## Authenticating with the Azure CLI
You will need an Azure account. Install the Azure CLI by running the following
(on Mac) and log in:

```
brew install azure-cli
az login
```

## Deploying Hail

Every resource in Azure must belong to a Resource Group. First, obtain
a resource group and make sure you have Owner permissions for that
resource group.

We will first create a Managed Identity which we will use later to
create resources through terraform. Run the following
(this only needs to be run once for a resource group, but it is ok to run it
more than once).

```
./create_terraform_identity.sh <RESOURCE_GROUP>
./bootstrap.sh setup_az <RESOURCE_GROUP> # FIXME use service principal
```

Create a `global.tfvars` file with the necessary variables
from $HAIL/infra/azure/variables.tf.

To setup and run the terraform, run

```
./bootstrap.sh run_azure_terraform <RESOURCE_GROUP>
./bootstrap.sh run_k8s_terraform <RESOURCE_GROUP>
```

Once terraform has completed successfully, note the `gateway_ip` in the
output and create an A record for the domain of your choosing for that
IP with a DNS provider.

## Bootstrap the cluster

We'll complete the rest of the process on a VM. To create one, run

```
./create_bootstrap_vm.sh <RESOURCE_GROUP>
```

SSH into the VM (ssh -i ~/.ssh/id_rsa <username>@<public_ip>).

Clone the hail repository:

```
git clone https://github.com/<repo_name>/hail.git
```

In the $HAIL/infra directory, run

```
./install_bootstrap_dependencies.sh
```

At this point, log out and ssh back in (so that changes to group settings
for Docker can be applied). In the $HAIL/infra/azure directory, run

```
./bootstrap.sh setup_az <RESOURCE_GROUP>
```

to download and authenticate with the azure CLI.

Run the following to authenticate docker and kubectl with the new
container registry and kubernetes cluster, respectively.

```
source $HAIL/devbin/functions.sh
azsetcluster <RESOURCE_GROUP>
```

Deploy unmanaged resources by running

```
./bootstrap.sh deploy_unmanaged
```

Build the batch worker image by running the following in $HAIL/batch:

```
./az-create-worker-image.sh <RESOURCE_GROUP> <REGION> <YOUR_USERNAME>
```

Finally, run the following to deploy Hail in the cluster.

```
download-secret global-config && sudo cp -r contents /global-config
download-secret zulip-config && sudo cp -r contents /zulip-config
cd ~/hail/infra/azure
./bootstrap.sh bootstrap <REPO>/hail:<BRANCH> deploy_batch
```
