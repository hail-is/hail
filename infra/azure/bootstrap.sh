#!/bin/bash

source ../bootstrap_utils.sh

setup_az() {
    curl --connect-timeout 5 \
         --max-time 10 \
         --retry 5 \
         --retry-all-errors \
         --retry-max-time 40 \
         --location \
         --fail \
         --silent \
         --show-error \
         https://aka.ms/InstallAzureCLIDeb | sudo bash
    az login --identity
}

setup_terraform_backend() {
    RESOURCE_GROUP=$1
    STORAGE_CONTAINER_NAME="tfstate"

    # NOTE: This name is assumed to be globally unique
    STORAGE_ACCOUNT_NAME="${RESOURCE_GROUP}hailterraform"

    # This storage account/container will be created if it does not already exist
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
    if [ -z "$1" ]; then
        echo "Arguments not supplied: <RESOURCE_GROUP>"
        return
    fi

    set -e
    setup_terraform_backend "$@"
    terraform apply -var-file=global.tfvars
}

post_terraform() {
    az ad app permission admin-consent --id "$(terraform output -raw auth_sp_application_id)"
}

$@
