#!/bin/bash

set -ex

if [ -z "$1" ] || [ -z  "$2" ] || [ -z "$3" ]; then
    echo "Usage: ./az-create-worker-image.sh <RESOURCE_GROUP> <REGION> <USERNAME>"
    exit 1
fi

RESOURCE_GROUP=$1
REGION=$2
USERNAME=$3

CONTAINER_REGISTRY_NAME=${RESOURCE_GROUP}
SHARED_GALLERY_NAME="${RESOURCE_GROUP}_batch"
BUILD_IMAGE_RESOURCE_GROUP="${RESOURCE_GROUP}-build-batch-worker-image"
VM_NAME=build-batch-worker-image
WORKER_VERSION=0.0.12

DOCKER_ROOT_IMAGE=$(kubectl get secret global-config \
    --template={{.data.docker_root_image}} \
    | base64 --decode)

SUBSCRIPTION_ID=$(az account list | jq -rj '.[0].id')
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
az group create --name $BUILD_IMAGE_RESOURCE_GROUP --location ${REGION}

az role assignment create \
    --resource-group $BUILD_IMAGE_RESOURCE_GROUP \
    --assignee ${BATCH_WORKER_PRINCIPAL_ID} \
    --role contributor

echo "Creating $VM_NAME VM..."

IP=$(az vm create \
    --resource-group $BUILD_IMAGE_RESOURCE_GROUP \
    --name $VM_NAME \
    --image UbuntuLTS \
    --generate-ssh-keys \
    --public-ip-sku Standard \
    --assign-identity ${BATCH_WORKER_IDENTITY} \
    | jq -jr '.publicIpAddress')

echo "$VM_NAME VM created successfully!"

python3 ../ci/jinja2_render.py "{\"global\":{\"container_registry_name\":\"${CONTAINER_REGISTRY_NAME}\",\"docker_root_image\":\"${DOCKER_ROOT_IMAGE}\"}}" build-batch-worker-image-startup-azure.sh build-batch-worker-image-startup-azure.sh.out

echo "Running image startup script..."

ssh -i '~/.ssh/id_rsa' \
    -o StrictHostKeyChecking="accept-new" \
    $USERNAME@$IP \
    'sudo bash -s ' < build-batch-worker-image-startup-azure.sh.out

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
    --gallery-image-definition batch-worker \
    --gallery-name ${SHARED_GALLERY_NAME} \
    --resource-group ${RESOURCE_GROUP} \
    --gallery-image-version ${WORKER_VERSION} || true

echo "Creating image..."

az sig image-version create \
    --resource-group ${RESOURCE_GROUP} \
    --gallery-name ${SHARED_GALLERY_NAME} \
    --gallery-image-definition batch-worker \
    --gallery-image-version ${WORKER_VERSION} \
    --target-regions ${REGION} \
    --replica-count 1 \
    --managed-image "/subscriptions/${SUBSCRIPTION_ID}/resourceGroups/${BUILD_IMAGE_RESOURCE_GROUP}/providers/Microsoft.Compute/virtualMachines/${VM_NAME}"

echo "Image created!"
echo "Deleting resource group $BUILD_IMAGE_RESOURCE_GROUP"

az group delete --name $BUILD_IMAGE_RESOURCE_GROUP --yes

echo "Resource group $BUILD_IMAGE_RESOURCE_GROUP deleted successfully!"
