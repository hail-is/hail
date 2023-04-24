#!/bin/bash

set -ex

source ../copy_image.sh

copy_if_not_present() {
    src_image=$1
    dest_image=$2
    if ! skopeo inspect "docker://docker.io/$1";
    then
        echo "$1 does not exist yet, doing nothing"
    elif skopeo inspect "docker://$2";
    then
      echo "$2 already exists, doing nothing"
    else
      echo "$2 does not exist, copying $1 to $2"
      copy_image $1 $2
    fi
}

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
    "python-dill:3.8"
    "python-dill:3.8-slim"
    "python-dill:3.9"
    "python-dill:3.9-slim"
    "python-dill:3.10"
    "python-dill:3.10-slim"
    "hail:${HAIL_PIP_VERSION}"
    "hailtop:${HAIL_PIP_VERSION}"
)
for image in "${images[@]}"
do
    copy_if_not_present "hailgenetics/${image}" "${DOCKER_PREFIX}/hailgenetics/${image}"
done
