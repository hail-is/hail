#!/bin/bash

set -ex

REGION="eastus"
RESOURCE_GROUP="hail-dev"
SHARED_GALLERY_NAME="batch"
SUBSCRIPTION_ID=`az account list | jq -rj '.[0].id'`

az vm deallocate --resource-group build-batch-worker-image --name build-batch-worker-image

az vm generalize --resource-group build-batch-worker-image --name build-batch-worker-image

az sig image-version delete \
    --gallery-image-definition batch-worker \
    --gallery-name ${SHARED_GALLERY_NAME} \
    --resource-group ${RESOURCE_GROUP} \
    --gallery-image-version 0.0.12 || true

az sig image-version create \
    --resource-group ${RESOURCE_GROUP} \
    --gallery-name ${SHARED_GALLERY_NAME} \
    --gallery-image-definition batch-worker \
    --gallery-image-version 0.0.12 \
    --target-regions ${REGION} \
    --replica-count 1 \
    --managed-image "/subscriptions/${SUBSCRIPTION_ID}/resourceGroups/build-batch-worker-image/providers/Microsoft.Compute/virtualMachines/build-batch-worker-image"

az group delete --name build-batch-worker-image
