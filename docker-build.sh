#!/bin/bash

set -ex

if [ -z "${NAMESPACE}" ]; then
    echo "Must specify a NAMESPACE environment variable"
    exit 1;
fi

CONTEXT="$(cd $1 && pwd)"
DOCKERFILE="$CONTEXT/$2"
REMOTE_IMAGE_NAME=$3

WITHOUT_TAG=$(echo $REMOTE_IMAGE_NAME | sed -E 's/(:[^:]+)(@[^@]+)?$//')
MAIN_CACHE=$(echo ${WITHOUT_TAG} | sed -E 's:/[^/]+$::')/cache

# DOCKER_BUILDKIT=1 docker build \
podman build \
       --platform linux/amd64 \
       --file ${DOCKERFILE} \
       --layers \
       --cache-from ${MAIN_CACHE} \
       --cache-to ${MAIN_CACHE} \
       --tag ${REMOTE_IMAGE_NAME} \
       ${CONTEXT}

# time DOCKER_BUILDKIT=1 docker push ${REMOTE_IMAGE_NAME}
time DOCKER_BUILDKIT=1 podman push ${REMOTE_IMAGE_NAME}
