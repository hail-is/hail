#!/bin/bash

set -ex

if [ -z "${NAMESPACE}" ]; then
    echo "Must specify a NAMESPACE environment variable"
    exit 1;
fi

CONTEXT="$(cd $1 && pwd)"
DOCKERFILE="$CONTEXT/$2"
REMOTE_IMAGE_NAME=$3
EXTRA_CACHE=$4

WITHOUT_TAG=$(echo $REMOTE_IMAGE_NAME | sed -E 's/(:[^:]+)(@[^@]+)?$//')
MAIN_CACHE=${WITHOUT_TAG}:cache

if [ "${NAMESPACE}" == "default" ]; then
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
       ${EXTRA_CACHE:+--cache-from ${EXTRA_CACHE}} \
       --build-arg BUILDKIT_INLINE_CACHE=1 \
       ${DOCKER_BUILD_ARGS} \
       --tag ${REMOTE_IMAGE_NAME} \
       --tag ${CACHE_IMAGE_NAME} \
       ${CONTEXT}

time DOCKER_BUILDKIT=1 docker push ${REMOTE_IMAGE_NAME}
time DOCKER_BUILDKIT=1 docker push ${CACHE_IMAGE_NAME}
