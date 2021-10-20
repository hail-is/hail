#!/bin/bash

set -ex

if [ -z "$1" ]; then
    echo "Usage: ./create_terraform_sp.sh <RESOURCE_GROUP>"
    exit 1
fi

RESOURCE_GROUP=$1
SUBSCRIPTION_ID=$(az account list | jq -rj '.[0].id')

TERRAFORM_PRINCIPAL_NAME=${RESOURCE_GROUP}-terraform
az ad sp create-for-rbac \
    --role="Owner" \
    --scopes="/subscriptions/${SUBSCRIPTION_ID}" \
    --name=${TERRAFORM_PRINCIPAL_NAME} \
    > terraform_principal.json
