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

python_dill_images=(
    "python-dill:3.7"
    "python-dill:3.7-slim"
    "python-dill:3.8"
    "python-dill:3.8-slim"
    "python-dill:3.9"
    "python-dill:3.9-slim"
    "python-dill:3.10"
    "python-dill:3.10-slim"
)

for image in "${python_dill_images[@]}"
do
    copy_image "hailgenetics/${image}" "${DOCKER_PREFIX}/hailgenetics/${image}"
done

pip_release_images=(
    "hail:${HAIL_PIP_VERSION}"
    "genetics:${HAIL_PIP_VERSION}"
)
for image in "${pip_release_images[@]}"
do
    copy_image "hailgenetics/${image}" "${DOCKER_PREFIX}/hailgenetics/${image}"
done
