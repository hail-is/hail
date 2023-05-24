#!/bin/bash

source ../bootstrap_utils.sh

function configure_gcloud() {
    ZONE=${1:-"us-central1-a"}
    REGION=${2:-"us"}

    # If you are using the Container Registry:
    # gcloud -q auth configure-docker
    gcloud -q auth configure-docker $REGION-docker.pkg.dev
    gcloud container clusters get-credentials --zone $ZONE vdc
}

"$@"
