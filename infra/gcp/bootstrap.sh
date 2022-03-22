#!/bin/bash

source ../bootstrap_utils.sh

function configure_gcloud() {
    ZONE=${1:-"us-central1-a"}

    gcloud -q auth configure-docker
    # If you are using the Artifact Registry:
    # gcloud -q auth configure-docker $REGION-docker.pkg.dev
    gcloud container clusters get-credentials --zone $ZONE vdc
}

"$@"
