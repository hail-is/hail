#!/bin/bash

set -ex

RESOURCE_GROUP=${1:-$RESOURCE_GROUP}
SUBSCRIPTION_ID=${2:-$SUBSCRIPTION_ID}

if [ -z "$RESOURCE_GROUP" -o -z "$SUBSCRIPTION_ID" ]; then
    cat <<EOF
./create_bootstrap_vm.sh <RESOURCE_GROUP> <SUBSCRIPTION_ID>

Either argument may be elided if an environment variable with the same name is
set. An empty string argument is interpreted as an elided argument.

If both environment variables and parameters are supplied, the parameters are
preferred.

Your subscription id may be found in the output of \`az account list\`.
EOF
    exit 1
fi

VM_NAME=bootstrap-vm-$(cat /dev/urandom | LC_ALL=C tr -dc 'a-z0-9' | head -c 5)

ip=$(az vm create \
    --name $VM_NAME \
    --resource-group $RESOURCE_GROUP \
    --image Canonical:0001-com-ubuntu-server-focal:20_04-lts:latest \
    --vnet-name default \
    --subnet k8s-subnet \
    --public-ip-sku Standard \
    --os-disk-size 200 \
    --generate-ssh-keys | jq -jr '.publicIpAddress')

az vm identity assign \
    --name $VM_NAME \
    --resource-group $RESOURCE_GROUP \
    --role Owner \
    --scope /subscriptions/$SUBSCRIPTION_ID

echo "Successfully created a vm. SSH into it with ssh -i ~/.ssh/id_rsa <username>@$ip"
