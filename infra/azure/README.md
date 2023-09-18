# Hail on Azure

This is a work in progress for setting up hail infrasture on Azure. The
following should be executed in the `$HAIL/infra/azure` directory unless
otherwise noted.

Prerequisites:

- You must have `jq`, `terraform` installed.
- Export `HAIL` as the root of the checked out Hail repository

## Authenticating with the Azure CLI

You will need an Azure account. Install the Azure CLI by running the following
(on Mac) and log in:

```
brew install azure-cli
az login
```

## Running Terraform

Every resource in Azure must belong to a Resource Group. First, obtain
a resource group and make sure you have Owner permissions for that
resource group.

You will also need a storage account and container for remotely storing
terraform state. The following command creates these and stores their names in
`remote_storage.tfvars`:

```
./bootstrap.sh create_terraform_remote_storage <RESOURCE_GROUP>
```

Initialize terraform:

```
./bootstrap.sh init_terraform <RESOURCE_GROUP>
```

Create a `global.tfvars` file with the necessary variables from
$HAIL/infra/azure/variables.tf.

Run terraform:

```
terraform apply -var-file=global.tfvars
```

Once terraform has completed successfully, you must create an A record for the
domain of your choosing pointing at the `gateway_ip` with a DNS provider. The
`gateway_ip` may be retrieved by executing the following command.

```
terraform output -raw gateway_ip
```

There is one resource which, unfortunately, cannot be configured directly with
Terraform. The service principal used by the `auth` service needs "admin
consent" to create new service accounts for new users. After you first run
terraform and whenever you recreate the auth service principal, you must grant
this new service principal admin consent:

```
./bootstrap.sh grant_auth_sp_admin_consent
```

## Bootstrap the cluster

We'll complete the rest of the process on a VM. To create one, run

```
./create_bootstrap_vm.sh <RESOURCE_GROUP> <SUBSCRIPTION_ID>
```

Find the public ip of the created bootstrap vm doing the following:

```
BOOTSTRAP_VM=$(az vm list -g hail | jq -r '.[].name')
PUBLIC_IP=$(az vm show -d -n $BOOTSTRAP_VM -g hail --query "publicIps" -o tsv)
echo $BOOTSTRAP_VM $PUBLIC_IP
```

SSH into the VM (ssh -i ~/.ssh/id_rsa <username>@$PUBLIC_IP).

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
./bootstrap.sh setup_az
```

to download and authenticate with the azure CLI.

Run the following to authenticate docker and kubectl with the new
container registry and kubernetes cluster, respectively.

```
azsetcluster <RESOURCE_GROUP>
```

The ACR authentication token only lasts three hours. If you pause the deployment
process after this point, you may have to rerun `azsetcluster` before continuing.

Edit `$HAIL/letsencrypt/subdomains.txt` to include just the services you plan
to use in this deployment, e.g. `auth`, `batch` and `batch-driver`.

Deploy unmanaged resources by running

```
./bootstrap.sh deploy_unmanaged
```

During the deploy while running into issues you may have to run the
above command multiple times. Each time it will try to create certificates
using letsencrypt. You may reach a limit on the number of attempts possible
withing a 24hr period, or it may fail if the specified certs already exist.
If this happens it will retrieve the exiting certs, but the deploy_unmanaged step will fail.

If the final letsencrypt step in deploy_unmanaged fails, you will have to
comment out the "set +x" line in letsencrypt.sh. Then set the environment
variable `DRY_RUN=1`. Re-run the deploy_unmanaged step again and copy the
kubectl secret from stdout.

Apply the secret manually using `kubectl apply` and revert the changes.
You can then move on from this step.


Build the batch worker image by running the following in $HAIL/batch:

```
./az-create-worker-image.sh
```

Finally, run the following to deploy Hail in the cluster.

```
download-secret global-config && sudo cp -r contents /global-config
download-secret database-server-config && sudo cp -r contents /sql-config
cd ~/hail/infra/azure
./bootstrap.sh bootstrap <REPO>/hail:<BRANCH> deploy_batch
```

Create the initial (developer) user. The OBJECT_ID is the Azure Active
Directory user's object ID.

You can find the current object id locally if you're logged in using:

```
az ad signed-in-user show | jq '.objectId'
```

```
./bootstrap.sh bootstrap <REPO>/hail:<BRANCH> create_initial_user <USERNAME> <OBJECT_ID>
```

Deploy the gateway:

```
make -C $HAIL/gateway envoy-xds-config deploy
```
