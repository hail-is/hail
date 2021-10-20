#!/bin/bash

set -ex

if [ -z  "$1" ]; then
    echo "Call like ./create_bootstrap_vm.sh <RESOURCE_GROUP>"
    exit 1
fi

RESOURCE_GROUP=$1

TERRAFORM_PRINCIPAL_ID=$(az identity show \
    --resource-group $RESOURCE_GROUP \
    --name terraform \
    --query id \
    --output tsv
)

ip=$(az vm create \
    --name bootstrap-vm \
    --resource-group $RESOURCE_GROUP \
    --image Canonical:UbuntuServer:18.04-LTS:latest \
    --vnet-name default \
    --subnet batch-worker-subnet \
    --public-ip-sku Standard \
    --assign-identity $TERRAFORM_PRINCIPAL_ID \
    --os-disk-size 200 \
    --generate-ssh-keys | jq -jr '.publicIpAddress')

echo "Successfully created a vm. SSH into it with ssh -i ~/.ssh/id_rsa <username>@$ip"
