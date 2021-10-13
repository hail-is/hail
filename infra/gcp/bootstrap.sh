#!/bin/bash

source ../bootstrap_utils.sh

function setup_gcloud() {
    ZONE=$1
    gcloud -q auth configure-docker
    # If you are using the Artifact Registry:
    # gcloud -q auth configure-docker $REGION-docker.pkg.dev
    gcloud container clusters get-credentials --zone $ZONE vdc
}

run_gcp_terraform() {
    terraform init
    terraform apply -var-file=$HOME/.hail/global.tfvars
}

run_k8s_terraform() {
    cat >../k8s/global.tfvars <<EOF
global_config = $(terraform output global_config)
sql_config    = $(terraform output sql_config)
EOF

    cd $HAIL/infra/k8s
    terraform init
    terraform apply -var-file=global.tfvars
    cd -
}


$@
