#!/bin/bash

set -ex

if [ -z  "$1" ]; then
    echo "Call like ./create_bootstrap_vm.sh <RESOURCE_GROUP>"
    exit 1
fi

RESOURCE_GROUP=$1

SUBSCRIPTION_ID=$(az account list | jq -rj '.[0].id')
VM_NAME=bootstrap-vm

ip=$(az vm create \
    --name $VM_NAME \
    --resource-group $RESOURCE_GROUP \
    --image Canonical:UbuntuServer:18.04-LTS:latest \
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
