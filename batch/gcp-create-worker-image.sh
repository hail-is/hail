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

# When you bump the WORKER_IMAGE_VERSION, you should also update:
# - the INSTANCE_VERSION in globals.py (add one to the current value)
# - the image name in batch/batch/cloud/gcp/driver/create_instance.py (should match this value)
WORKER_IMAGE_VERSION=22

if [ "$NAMESPACE" == "default" ]; then
    WORKER_IMAGE=batch-worker-${WORKER_IMAGE_VERSION}
    BUILDER=build-batch-worker-image
else
    WORKER_IMAGE=batch-worker-$NAMESPACE-${WORKER_IMAGE_VERSION}
    BUILDER=build-batch-worker-$NAMESPACE-image
fi

UBUNTU_IMAGE=ubuntu-minimal-2404-noble-amd64-v20260704

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

wait_for_vm() {
    local -a frames=('⠋' '⠙' '⠹' '⠸' '⠼' '⠴' '⠦' '⠧' '⠇' '⠏')
    local i=0 last_poll=0 vm_status='RUNNING'
    local start=$SECONDS

    while [ "$vm_status" == "RUNNING" ]; do
        if (( SECONDS - last_poll >= 5 )); then
            vm_status=$(gcloud compute instances describe "$BUILDER" \
                --project "$PROJECT" --zone "$ZONE" --format='value(status)' 2>/dev/null) || true
            last_poll=$SECONDS
        fi
        local elapsed=$(( SECONDS - start ))
        printf '\r  %s %s [%s] %dm %02ds  ' \
            "${frames[i % ${#frames[@]}]}" "$BUILDER" "$vm_status" \
            "$(( elapsed / 60 ))" "$(( elapsed % 60 ))"
        i=$(( i + 1 ))
        sleep 0.1
    done

    local elapsed=$(( SECONDS - start ))
    printf '\r  ✓ %s done in %dm %02ds%30s\n' \
        "$BUILDER" "$(( elapsed / 60 ))" "$(( elapsed % 60 ))" ''
}

main() {
    set -x
    create_build_image_instance
    set +x
    wait_for_vm
    set -x
    create_worker_image
}

confirm "Building image $WORKER_IMAGE with properties:\n Version: ${WORKER_IMAGE_VERSION}\n Project: ${PROJECT}\n Zone: ${ZONE}" && main
