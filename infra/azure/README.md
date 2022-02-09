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
./bootstrap.sh setup_az
```

to download and authenticate with the azure CLI.

Run the following to authenticate docker and kubectl with the new
container registry and kubernetes cluster, respectively.

```
azsetcluster <RESOURCE_GROUP>
```

Deploy unmanaged resources by running

```
./bootstrap.sh deploy_unmanaged
```

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

```
./bootstrap.sh bootstrap <REPO>/hail:<BRANCH> create_initial_user <USERNAME> <OBJECT_ID>
```

Deploy the gateway service. First trim down `$HAIL/letsencrypt/subdomains.txt`
to only the services that are deployed and then run

```
make -C $HAIL/gateway deploy
```
