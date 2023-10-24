#!/bin/bash

set -ex

CONTEXT="$(cd $1 && pwd)"
DOCKERFILE="$CONTEXT/$2"
IMAGE_NAME=$3

WITHOUT_TAG=$(echo $IMAGE_NAME | sed -E 's/(:[^:]+)(@[^@]+)?$//')
MAIN_CACHE=${WITHOUT_TAG}:cache

if [ "${NAMESPACE:-default}" == "default" ]; then
    CACHE_IMAGE_NAME=${MAIN_CACHE}
else
    DEV_CACHE=${WITHOUT_TAG}:cache-${NAMESPACE}
    CACHE_IMAGE_NAME=${DEV_CACHE}
fi

DOCKER_BUILDKIT=1 docker build \
       --platform linux/amd64 \
       --file ${DOCKERFILE} \
       --cache-from ${MAIN_CACHE} \
       ${DEV_CACHE:+--cache-from ${DEV_CACHE}} \
       --build-arg BUILDKIT_INLINE_CACHE=1 \
       ${DOCKER_BUILD_ARGS} \
       --tag ${IMAGE_NAME} \
       --tag ${CACHE_IMAGE_NAME} \
       ${CONTEXT}
