#!/bin/bash

set -e

cd "$(dirname "$0")"
source ../devbin/functions.sh

if [ -z "${NAMESPACE}" ]; then
    echo "Must specify a NAMESPACE environment variable"
    exit 1;
fi

REGION=us-east-1  #$(get_global_config_field aws_region $NAMESPACE)
DOCKER_ROOT_IMAGE=$(get_global_config_field docker_root_image $NAMESPACE)

# When you bump the WORKER_IMAGE_VERSION, you should also update:
# - the INSTANCE_VERSION in globals.py (add one to the current value)
# - the image name in batch/batch/cloud/gcp/driver/create_instance.py (should match this value)
WORKER_IMAGE_VERSION=33

if [ "$NAMESPACE" == "default" ]; then
    WORKER_IMAGE=batch-worker-${WORKER_IMAGE_VERSION}
    BUILDER=build-batch-worker-image
else
    WORKER_IMAGE=batch-worker-$NAMESPACE-${WORKER_IMAGE_VERSION}
    BUILDER=build-batch-worker-$NAMESPACE-image
fi

UBUNTU_IMAGE=/aws/service/canonical/ubuntu/server-minimal/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id

WORKER_IMAGE_EXISTS=false
if [[ "$(aws ec2 describe-images --filter="Name=tag:Name,Values=${WORKER_IMAGE}" --query="Reservations[*].Instances[*].[InstanceId]" --region "${REGION}")" -ne "null" ]]; then
    WORKER_IMAGE_EXISTS=true
    if [ "$NAMESPACE" == "default" ]; then
        echo "ERROR: Image $WORKER_IMAGE already exists in region $REGION. Delete it first or bump WORKER_IMAGE_VERSION."
        exit 1
    else
        echo "WARNING: Image $WORKER_IMAGE already exists in region $REGION and will be overwritten."
    fi
fi

LEFTOVER_BUILDERS="$(aws ec2 describe-instances --region $REGION --filter="Name=tag:Name,Values=${BUILDER}" --query="Reservations[*].Instances[*].[InstanceId]" --output text | tr '\n' ' ')"
if [[ -n "$LEFTOVER_BUILDERS" ]]; then
    echo "WARNING: Found leftover builder VM(s) in region $REGION:"
    echo "$LEFTOVER_BUILDERS"
fi

create_build_image_instance() {
    if [[ -n "$LEFTOVER_BUILDERS" ]]; then
        aws ec2 terminate-instances --region $REGION --instance-ids ${LEFTOVER_BUILDERS}
    fi

    python3 ../ci/jinja2_render.py '{"global":{"docker_root_image":"'${DOCKER_ROOT_IMAGE}'"}}' \
        build-batch-worker-image-startup-aws.sh build-batch-worker-image-startup-aws.sh.out

    UBUNTU_AMI_ID=$(aws ssm get-parameter --name "$UBUNTU_IMAGE" --query="Parameter.Value" --output text)

    BUILDER_ID=$(aws ec2 run-instances \
        --image-id ${UBUNTU_AMI_ID} \
        --count 1 \
        --instance-type t2.large \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value='${BUILDER}'}]" \
        --user-data file://build-batch-worker-image-startup-aws.sh.out \
	--region ${REGION} \
        --query "Instances[*].[InstanceId]" \
	--output text
    )
}

create_worker_image() {
    if [ "$WORKER_IMAGE_EXISTS" == "true" ]; then
        aws ec2 deregister-image --image-id $WORKER_IMAGE --region $REGION
    fi

    aws ec2 create-image \
        --instance-id ${BUILDER_ID} \
        --name ${WORKER_IMAGE} \
        --region ${REGION}

    aws ec2 terminate-instances \
        --instance-ids ${BUILDER_ID} \
        --region ${REGION}
}

main() {
    set -x
    create_build_image_instance
    while [ "$(aws ec2 describe-instance-status --instance-ids ${BUILDER_ID} --include-all-instances --query='InstanceStatuses[0].InstanceState.Name' --output text)" != "stopped" ];
    do
        sleep 5
    done
    create_worker_image
}

confirm "Building image $WORKER_IMAGE with properties:\n Version: ${WORKER_IMAGE_VERSION}\n Region: ${REGION}" && main
