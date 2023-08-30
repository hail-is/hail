#!/bin/bash

set -ex

source $HAIL/devbin/functions.sh

download-secret global-config
SUBSCRIPTION_ID=$(cat contents/azure_subscription_id)
RESOURCE_GROUP=$(cat contents/azure_resource_group)
LOCATION=$(cat contents/azure_location)
DOCKER_PREFIX=$(cat contents/docker_prefix)
DOCKER_ROOT_IMAGE=$(cat contents/docker_root_image)
popd

SHARED_GALLERY_NAME="${RESOURCE_GROUP}_batch"
BUILD_IMAGE_RESOURCE_GROUP="${RESOURCE_GROUP}-build-batch-worker-image"
VM_NAME=build-batch-worker-image
WORKER_VERSION=0.0.13

USERNAME=$(whoami)

BATCH_WORKER_PRINCIPAL_ID=$(az identity show \
    --resource-group ${RESOURCE_GROUP} \
    --name batch-worker \
    --query principalId \
    --output tsv)
BATCH_WORKER_IDENTITY=$(az identity show \
    --resource-group ${RESOURCE_GROUP} \
    --name batch-worker \
    --query id \
    --output tsv)

echo "Creating $BUILD_IMAGE_RESOURCE_GROUP resource group..."

az group delete --name $BUILD_IMAGE_RESOURCE_GROUP --yes || true
az group create --name $BUILD_IMAGE_RESOURCE_GROUP --location ${LOCATION}

az role assignment create \
    --resource-group $BUILD_IMAGE_RESOURCE_GROUP \
    --assignee ${BATCH_WORKER_PRINCIPAL_ID} \
    --role contributor

echo "Creating $VM_NAME VM..."

IP=$(az vm create \
    --resource-group $BUILD_IMAGE_RESOURCE_GROUP \
    --name $VM_NAME \
    --image Ubuntu2204 \
    --generate-ssh-keys \
    --public-ip-sku Standard \
    --assign-identity ${BATCH_WORKER_IDENTITY} \
    | jq -jr '.publicIpAddress')

echo "$VM_NAME VM created successfully!"

python3 ../ci/jinja2_render.py "{\"global\":{\"docker_prefix\":\"${DOCKER_PREFIX}\",\"docker_root_image\":\"${DOCKER_ROOT_IMAGE}\"}}" build-batch-worker-image-startup-azure.sh build-batch-worker-image-startup-azure.sh.out

echo "Running image startup script..."

az vm run-command invoke \
    --resource-group $BUILD_IMAGE_RESOURCE_GROUP \
    --command-id RunShellScript \
    --name $VM_NAME \
    --scripts "@build-batch-worker-image-startup-azure.sh.out"

echo "Startup script completed!"
echo "Shutting down agent..."
ssh -i '~/.ssh/id_rsa' \
    -o StrictHostKeyChecking="accept-new" \
    $USERNAME@$IP \
    'sudo waagent -deprovision+user -force'

echo "Deprovisioned agent"

az vm deallocate \
    --resource-group $BUILD_IMAGE_RESOURCE_GROUP \
    --name $VM_NAME

az vm generalize \
    --resource-group $BUILD_IMAGE_RESOURCE_GROUP \
    --name $VM_NAME

az sig image-version delete \
    --gallery-image-definition batch-worker-22-04 \
    --gallery-name ${SHARED_GALLERY_NAME} \
    --resource-group ${RESOURCE_GROUP} \
    --gallery-image-version ${WORKER_VERSION} || true

echo "Creating image..."

az sig image-version create \
    --resource-group ${RESOURCE_GROUP} \
    --gallery-name ${SHARED_GALLERY_NAME} \
    --gallery-image-definition batch-worker-22-04 \
    --gallery-image-version ${WORKER_VERSION} \
    --target-regions ${LOCATION} \
    --replica-count 1 \
    --managed-image "/subscriptions/${SUBSCRIPTION_ID}/resourceGroups/${BUILD_IMAGE_RESOURCE_GROUP}/providers/Microsoft.Compute/virtualMachines/${VM_NAME}"

echo "Image created!"
echo "Deleting resource group $BUILD_IMAGE_RESOURCE_GROUP"

az group delete --name $BUILD_IMAGE_RESOURCE_GROUP --yes

echo "Resource group $BUILD_IMAGE_RESOURCE_GROUP deleted successfully!"
