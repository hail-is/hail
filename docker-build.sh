#!/bin/bash

CONTEXT="$(cd $1 && pwd)"
DOCKERFILE="$CONTEXT/$2"
REMOTE_IMAGE_NAME=$3
EXTRA_CACHE=$4

CACHE_NAME=$(echo $REMOTE_IMAGE_NAME | sed -E 's/(:[^:]+)(@[^@]+)?$//'):cache

DOCKER_BUILDKIT=1 docker build \
       --file ${DOCKERFILE} \
       --cache-from ${CACHE_NAME} \
       ${EXTRA_CACHE:+--cache-from ${EXTRA_CACHE}} \
       --build-arg BUILDKIT_INLINE_CACHE=1 \
       --tag ${REMOTE_IMAGE_NAME} \
       --tag ${CACHE_NAME} \
       ${CONTEXT}

time DOCKER_BUILDKIT=1 docker push ${REMOTE_IMAGE_NAME}
time DOCKER_BUILDKIT=1 docker push ${CACHE_NAME}











