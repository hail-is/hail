#!/bin/bash

set -ex

source copy_image.sh

copy_if_not_present() {
    src_image=$1
    dest_image=$2
    if ! skopeo inspect "docker://docker.io/$1";
    then
        echo "$1 does not exist yet, doing nothing"
    else
      echo "copying $1 to $2"
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
    "hailgenetics/python-dill:3.9"
    "hailgenetics/python-dill:3.9-slim"
    "hailgenetics/python-dill:3.10"
    "hailgenetics/python-dill:3.10-slim"
    "hailgenetics/python-dill:3.11"
    "hailgenetics/python-dill:3.11-slim"
    "hailgenetics/hail:${HAIL_PIP_VERSION}"
    "hailgenetics/hailtop:${HAIL_PIP_VERSION}"
    "hailgenetics/vep-grch37-85:${HAIL_PIP_VERSION}"
    "hailgenetics/vep-grch38-95:${HAIL_PIP_VERSION}"
    "ubuntu:20.04"
    "ubuntu:22.04"
    "ubuntu:24.04"
)
for image in "${images[@]}"
do
    copy_if_not_present "${image}" "${DOCKER_PREFIX}/${image}"
done
