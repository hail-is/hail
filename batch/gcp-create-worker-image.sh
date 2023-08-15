#!/bin/bash

set -e

cd "$(dirname "$0")"
source ../devbin/functions.sh

if [ -z "${NAMESPACE}" ]; then
    echo "Must specify a NAMESPACE environment variable"
    exit 1;
fi

PROJECT=$(get_global_config_field gcp_project $NAMESPACE)
ZONE=$(get_global_config_field gcp_zone $NAMESPACE)
DOCKER_ROOT_IMAGE=$(get_global_config_field docker_root_image $NAMESPACE)

WORKER_IMAGE_VERSION=12

if [ "$NAMESPACE" == "default" ]; then
    WORKER_IMAGE=batch-worker-${WORKER_IMAGE_VERSION}
    BUILDER=build-batch-worker-image
else
    WORKER_IMAGE=batch-worker-$NAMESPACE-${WORKER_IMAGE_VERSION}
    BUILDER=build-batch-worker-$NAMESPACE-image
fi

UBUNTU_IMAGE=ubuntu-minimal-2204-jammy-v20230726

create_build_image_instance() {
    gcloud -q compute --project ${PROJECT} instances delete \
        --zone=${ZONE} ${BUILDER} || true

    python3 ../ci/jinja2_render.py '{"global":{"docker_root_image":"'${DOCKER_ROOT_IMAGE}'"}}' \
        build-batch-worker-image-startup-gcp.sh build-batch-worker-image-startup-gcp.sh.out

    gcloud -q compute instances create ${BUILDER} \
        --project ${PROJECT}  \
        --zone=${ZONE} \
        --machine-type=n1-standard-1 \
        --network=default \
        --subnet=default \
        --network-tier=PREMIUM \
        --metadata-from-file startup-script=build-batch-worker-image-startup-gcp.sh.out \
        --no-restart-on-failure \
        --maintenance-policy=MIGRATE \
        --scopes=https://www.googleapis.com/auth/cloud-platform \
        --image=${UBUNTU_IMAGE} \
        --image-project=ubuntu-os-cloud \
        --boot-disk-size=10GB \
        --boot-disk-type=pd-ssd
}

create_worker_image() {
    gcloud -q compute images delete $WORKER_IMAGE \
        --project ${PROJECT} || true

    gcloud -q compute images create $WORKER_IMAGE \
        --project ${PROJECT} \
        --source-disk-zone=${ZONE} \
        --source-disk=${BUILDER}

    gcloud -q compute instances delete ${BUILDER} \
        --project ${PROJECT} \
        --zone=${ZONE}
}

main() {
    set -x
    create_build_image_instance
    while [ "$(gcloud compute instances describe ${BUILDER} --project ${PROJECT} --zone ${ZONE} --format='value(status)')" == "RUNNING" ];
    do
        sleep 5
    done
    create_worker_image
}

confirm "Building image $WORKER_IMAGE with properties:\n Version: ${WORKER_IMAGE_VERSION}\n Project: ${PROJECT}\n Zone: ${ZONE}" && main
