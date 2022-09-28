#!/bin/bash

set -ex

source ../copy_image.sh

if [[ -z "${DOCKER_PREFIX}" ]];
then
    echo "Env variable DOCKER_PREFIX must be set"
    exit 1
fi

if [[ -z "${HAIL_PIP_VERSION}" ]];
then
    echo "Env variable HAIL_PIP_VERSION must be set"
    exit 1
fi

images=(
    "python-dill:3.7"
    "python-dill:3.7-slim"
    "python-dill:3.8"
    "python-dill:3.8-slim"
    "python-dill:3.9"
    "python-dill:3.9-slim"
    "python-dill:3.10"
    "python-dill:3.10-slim"
    "hail:${HAIL_PIP_VERSION}"
    "genetics:${HAIL_PIP_VERSION}"
)
for image in "${images[@]}"
do
    copy_image "hailgenetics/${image}" "${DOCKER_PREFIX}/hailgenetics/${image}"
done
