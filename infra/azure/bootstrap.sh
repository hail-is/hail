#!/bin/bash

source ../bootstrap_utils.sh

setup_az() {
    curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
    az login --identity
}

export_terraform_vars() {
    export ARM_SUBSCRIPTION_ID=$(az account list | jq -rj '.[0].id')
    export ARM_TENANT_ID=$(az account list | jq -rj '.[0].tenantId')
    export ARM_CLIENT_ID=$(jq -rj '.appId' $HAIL/infra/azure/terraform_principal.json)
    export ARM_CLIENT_SECRET=$(jq -rj '.password' $HAIL/infra/azure/terraform_principal.json)
}

setup_terraform_backend() {
    RESOURCE_GROUP=$1
    STORAGE_CONTAINER_NAME=$2

    # This storage account will be created if it does not already exist
    # Assuming this name is globally unique
    STORAGE_ACCOUNT_NAME="${RESOURCE_GROUP}hailterraform"

    az storage account create -n $STORAGE_ACCOUNT_NAME -g $RESOURCE_GROUP
    STORAGE_ACCOUNT_KEY=$(az storage account keys list \
        -g $RESOURCE_GROUP \
        --account-name $STORAGE_ACCOUNT_NAME \
        | jq -rj '.[0].value'
    )

    az storage container create -n $STORAGE_CONTAINER_NAME \
        --account-name $STORAGE_ACCOUNT_NAME \
        --account-key $STORAGE_ACCOUNT_KEY

    cat >backend-config.tfvars <<EOF
storage_account_name = "$STORAGE_ACCOUNT_NAME"
container_name       = "$STORAGE_CONTAINER_NAME"
access_key           = "$STORAGE_ACCOUNT_KEY"
key                  = "haildev.tfstate"
EOF

    terraform init -backend-config=backend-config.tfvars
}

run_terraform() {
    if [ -z "$1" ] || [ -z "$2" ]; then
        echo "Arguments not supplied: <RESOURCE_GROUP> <STORAGE_CONTAINER_NAME>"
        return
    fi

    setup_terraform_backend "$@"
    terraform apply -var-file=global.tfvars
}

run_azure_terraform() {
    if [ -z "$1" ]; then
        echo "Usage: run_azure_terraform <RESOURCE_GROUP>"
        return
    fi
    run_terraform "$1" azure
}

run_k8s_terraform() {
    if [ -z "$1" ]; then
        echo "Usage: run_k8s_terraform <RESOURCE_GROUP>"
        return
    fi

    terraform output -raw kube_config >$HOME/.kube_config
    cat >../k8s/global.tfvars <<EOF
global_config = $(terraform output global_config)
sql_config    = $(terraform output sql_config)
acr_push_credentials = $(terraform output acr_push_credentials)
service_credentials = $(terraform output service_credentials)
EOF

    cd $HAIL/infra/k8s
    run_terraform "$1" k8s
    cd -
}

$@
