#!/bin/bash

source ../bootstrap_utils.sh

function configure_gcloud() {
    ZONE=${1:-"australia-southeast1-b"}
    REGION=${2:-"australia-southeast1"}

    gcloud -q auth configure-docker $REGION-docker.pkg.dev
    gcloud container clusters get-credentials --zone $ZONE vdc
}

"$@"
