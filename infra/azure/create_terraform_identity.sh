#!/bin/bash

set -ex

RESOURCE_GROUP=$1
SUBSCRIPTION_ID=$(az account list | jq -rj '.[0].id')

TERRAFORM_PRINCIPAL_ID=$(az identity create \
    --name terraform \
    --resource-group $RESOURCE_GROUP \
    --subscription $SUBSCRIPTION_ID | jq -rj '.principalId')

az role assignment create \
    --assignee $TERRAFORM_PRINCIPAL_ID \
    --role Owner \
    --scope /subscriptions/$SUBSCRIPTION_ID
