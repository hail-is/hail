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

TERRAFORM_APP_ID=$(cat terraform_principal.json | jq -jr '.appId')

GRAPH_API_GUID="00000003-0000-0000-c000-000000000000"
APPLICATION_READ_WRITE_ALL="bdfbf15f-ee85-4955-8675-146e8e5296b5"

az ad app permission add \
    --id $TERRAFORM_APP_ID \
    --api ${GRAPH_API_GUID} \
    --api-permissions ${APPLICATION_READ_WRITE_ALL}=Role

# Grant the permissions requested above
az ad app permission admin-consent \
    --id $TERRAFORM_APP_ID
