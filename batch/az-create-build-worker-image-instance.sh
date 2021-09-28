#!/bin/bash

set -ex

REGION="eastus"
RESOURCE_GROUP="hail-dev"
CONTAINER_REGISTRY_NAME=${RESOURCE_GROUP}
DOCKER_ROOT_IMAGE="${CONTAINER_REGISTRY_NAME}.azurecr.io/ubuntu:18.04"
BATCH_WORKER_PRINCIPAL_ID=`az identity show --resource-group ${RESOURCE_GROUP} --name batch-worker --query principalId --output tsv`
BATCH_WORKER_IDENTITY=`az identity show --resource-group ${RESOURCE_GROUP} --name batch-worker --query id --output tsv`
SSH_PUBLIC_KEY=${SSH_PUBLIC_KEY:="~/.ssh/id_rsa.pub"}

az group delete --name build-batch-worker-image || true
az group create --name build-batch-worker-image --location ${REGION}

az role assignment create --assignee ${BATCH_WORKER_PRINCIPAL_ID} --role contributor -g build-batch-worker-image

az vm create --resource-group build-batch-worker-image \
    --name build-batch-worker-image \
    --image UbuntuLTS \
    --ssh-key-values ${SSH_PUBLIC_KEY} \
    --public-ip-sku Standard \
    --assign-identity ${BATCH_WORKER_IDENTITY}

python3 ../ci/jinja2_render.py "{\"global\":{\"container_registry_name\":\"${CONTAINER_REGISTRY_NAME}\",\"docker_root_image\":\"${DOCKER_ROOT_IMAGE}\"}}" build-batch-worker-image-startup-azure.sh build-batch-worker-image-startup-azure.sh.out

az vm run-command invoke --command-id RunShellScript --name build-batch-worker-image --resource-group build-batch-worker-image --scripts "@build-batch-worker-image-startup-azure.sh.out"
