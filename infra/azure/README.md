# Hail on Azure

This is an incomplete, work in progress for setting up hail infrasture on Azure. The
following should all be executed in the `$HAIL/infra/azure` directory.

First, set some important environment variables that we will need down the road:

```
export AZ_RESOURCE_GROUP_NAME=<resource_group_name>
```

## Authenticating with the Azure CLI
You will need an Azure account. Install the Azure CLI by running the following (on Mac) and log in:

```
brew install azure-cli
az login
```

Every resource created in Azure must belong to a subscription.
Run the following to set an environment variable for the subscription id we will
use in later steps (assuming you only belong to one subscription or you intend to
use the first subscription listed).
Note: these environment variables prefixed with `ARM_` are used by terraform
later so be sure to use the same names.

```
export ARM_SUBSCRIPTION_ID=$(az account list | jq -rj '.[0].id')
export ARM_TENANT_ID=$(az account list | jq -rj '.[0].tenantId')
```

Next, we will create a service principal with the Contributor role for terraform
to use. If the service principal already exists, you can retrieve the client ID
and client secret credentials from another developer who has them, or reset
them. Either to create the service principal or reset it, run the following

```
az ad sp create-for-rbac --role="Contributor" --scopes="/subscriptions/${ARM_SUBSCRIPTION_ID}" --name="terraform-principal" > terraform_principal.json
```

Note: Your developer Azure account must be an owner of the terraform principal service principal.
To grant ownership to another developer, go in the Azure portal to
Azure Active Directory > App registrations > terraform-principal > Owners.

and then run the following block to log in using those credentials. It is not
possible to programmatically retrieve the SP credentials [without resetting
them](https://stackoverflow.com/questions/60535578/how-do-i-retrieve-the-service-principal-password-after-creation-using-the-azure/60537958).
Note: it might take a minute for the SP credentials to propagate.
If the folllowing fails, wait a moment and try again.

```
export ARM_CLIENT_ID=$(jq -rj '.appId' terraform_principal.json)
export ARM_CLIENT_SECRET=$(jq -rj '.password' terraform_principal.json)
az login --service-principal -u $ARM_CLIENT_ID -p $ARM_CLIENT_SECRET --tenant $ARM_TENANT_ID
```

## Setting up Terraform Remote State Management

Unless you're running a particularly lonely ship, you will want Terraform state to
be shared so that different developers can have a single consistent
view of the state of the infrastructure. The following resources only need to be created
once per resource group. If they already exist, you still need to define the environment
variables in this section, but you do not need to run any `create` commands.

Create a storage account that will own the storage container holding terraform state.
Note you might need to fiddle with the account name to satisfy the azure naming restrictions.

```
# This name must be alphanumeric and globally unique
export STORAGE_ACCOUNT_NAME=haildevterraform
az storage account create -n $STORAGE_ACCOUNT_NAME -g $AZ_RESOURCE_GROUP_NAME
export STORAGE_ACCOUNT_KEY=$(az storage account keys list --resource-group $AZ_RESOURCE_GROUP_NAME --account-name $STORAGE_ACCOUNT_NAME | jq -rj '.[0].value')
```

Create the storage container that terraform will use to store its state.

```
export STORAGE_CONTAINER_NAME=tfstate
az storage container create -n $STORAGE_CONTAINER_NAME --account-name $STORAGE_ACCOUNT_NAME --account-key $STORAGE_ACCOUNT_KEY
```

If you see an output of
```
{
  "created": true
}
```
the create step was successful. Finally, create the backend config file used to initialize terraform.

```
cat >backend-config.tfvars <<EOF
storage_account_name = "$STORAGE_ACCOUNT_NAME"
container_name       = "$STORAGE_CONTAINER_NAME"
access_key           = "$STORAGE_ACCOUNT_KEY"
key                  = "haildev.tfstate"
EOF
```

## Running Terraform

Initialize terraform. If you do not with to track terraform state remotely, you can
forego the backend-config parameters.

```
terraform init -backend-config=backend-config.tfvars
```

Next, create a `global.tfvars` file from the following template and replace in the relevant fields.

```
az_resource_group_name = "<resource_group_name>"
```

Run `terraform apply -var-file=global.tfvars`. To sync the kubernetes config, run

```
az aks get-credentials --name vdc --resource-group $AZ_RESOURCE_GROUP_NAME
```

you can now use `kubectl` to communicate with AKS cluster.
