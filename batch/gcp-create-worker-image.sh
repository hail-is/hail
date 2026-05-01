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

WORKER_IMAGE_VERSION=19

if [ "$NAMESPACE" == "default" ]; then
    WORKER_IMAGE=batch-worker-${WORKER_IMAGE_VERSION}
    BUILDER=build-batch-worker-image
else
    WORKER_IMAGE=batch-worker-$NAMESPACE-${WORKER_IMAGE_VERSION}
    BUILDER=build-batch-worker-$NAMESPACE-image
fi

UBUNTU_IMAGE=ubuntu-minimal-2404-noble-amd64-v20260429

WORKER_IMAGE_EXISTS=false
if [[ -n "$(gcloud compute images list --project "${PROJECT}" --filter="name=${WORKER_IMAGE}" --format='value(name)')" ]]; then
    WORKER_IMAGE_EXISTS=true
    if [ "$NAMESPACE" == "default" ]; then
        echo "ERROR: Image $WORKER_IMAGE already exists in project $PROJECT. Delete it first or bump WORKER_IMAGE_VERSION."
        exit 1
    else
        echo "WARNING: Image $WORKER_IMAGE already exists in project $PROJECT and will be overwritten."
    fi
fi

LEFTOVER_BUILDERS="$(gcloud compute instances list --project "${PROJECT}" --filter="name~^build-batch-worker" --format='value(name,zone)')"
if [[ -n "$LEFTOVER_BUILDERS" ]]; then
    echo "WARNING: Found leftover builder VM(s) in project $PROJECT:"
    echo "$LEFTOVER_BUILDERS"
fi
BUILDER_EXISTS="$(echo "$LEFTOVER_BUILDERS" | grep -c "^${BUILDER}\b" || true)"

create_build_image_instance() {
    if [[ "$BUILDER_EXISTS" -gt 0 ]]; then
        gcloud -q compute --project ${PROJECT} instances delete --zone=${ZONE} ${BUILDER}
    fi

    python3 ../ci/jinja2_render.py '{"global":{"docker_root_image":"'${DOCKER_ROOT_IMAGE}'"}}' \
        build-batch-worker-image-startup-gcp.sh build-batch-worker-image-startup-gcp.sh.out

    gcloud -q compute instances create ${BUILDER} \
        --project ${PROJECT}  \
        --zone=${ZONE} \
        --machine-type=n1-standard-4 \
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
    if [ "$WORKER_IMAGE_EXISTS" == "true" ]; then
        gcloud -q compute images delete $WORKER_IMAGE --project ${PROJECT}
    fi

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
