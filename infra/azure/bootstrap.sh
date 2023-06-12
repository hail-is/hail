#!/bin/bash

source ../bootstrap_utils.sh

setup_az() {
    curl --connect-timeout 5 \
         --max-time 10 \
         --retry 5 \
         --retry-max-time 40 \
         --location \
         --fail \
         --silent \
         --show-error \
         https://aka.ms/InstallAzureCLIDeb | sudo bash
    az login --identity
}

create_terraform_remote_storage() {
    RESOURCE_GROUP=${RESOURCE_GROUP:-$1}
    STORAGE_CONTAINER_NAME=${STORAGE_CONTAINER_NAME:-"tfstate"}

    # NOTE: This name is assumed to be globally unique
    # NOTE: must be 3-24 lowercase letters and numbers only
    possibly_invalid_storage_account_name="$(cat /dev/urandom | LC_ALL=C tr -dc '0-9' | head -c 4)${RESOURCE_GROUP}"
    STORAGE_ACCOUNT_NAME=$(LC_ALL=C tr -dc 'a-z0-9' <<< "${possibly_invalid_storage_account_name}" | head -c 24)

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
    cat >remote_storage.tfvars <<EOF
resource_group_name  = "$RESOURCE_GROUP"
storage_account_name = "$STORAGE_ACCOUNT_NAME"
container_name       = "$STORAGE_CONTAINER_NAME"
key                  = "haildev.tfstate"
EOF
}

init_terraform() {
    RESOURCE_GROUP=${RESOURCE_GROUP:-$1}
    ARGS=${@:2}

    STORAGE_ACCOUNT_NAME=$(cat remote_storage.tfvars | grep "storage_account_name" | awk '{ print $3 }' | tr -d '"')
    STORAGE_ACCOUNT_KEY=$(az storage account keys list \
        -g $RESOURCE_GROUP \
        --account-name $STORAGE_ACCOUNT_NAME \
        | jq -rj '.[0].value'
    )
    remote_storage_access_key="$(mktemp)"
    trap 'rm -f -- "$remote_storage_access_key"' EXIT

    cat >$remote_storage_access_key <<EOF
access_key           = "$STORAGE_ACCOUNT_KEY"
EOF

    terraform init -backend-config=$remote_storage_access_key -backend-config=remote_storage.tfvars $ARGS
}

grant_auth_sp_admin_consent() {
    az ad app permission admin-consent --id "$(terraform output -raw auth_sp_application_id)"
}

$@
