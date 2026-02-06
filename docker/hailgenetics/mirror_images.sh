#!/bin/bash

set -ex

source ../copy_image.sh

copy_if_not_present() {
    src_image=$1
    dest_image=$2
    # If DOCKERHUB_PREFIX is set, prepend it to the source image
    if [ -n "${DOCKERHUB_PREFIX}" ]
    then
        # Image is already namespaced (e.g., "hailgenetics/python-dill:3.10")
        src="${DOCKERHUB_PREFIX}/${src_image}"
    else
        # No dockerhub prefix, use docker.io as before
        src="$src_image"
    fi
    
    # Check if image exists (using docker.io for the check if no prefix, otherwise use the prefixed source)
    if [ -n "${DOCKERHUB_PREFIX}" ]
    then
        check_src="$src"
    else
        check_src="docker.io/$src_image"
    fi
    
    if ! skopeo inspect "docker://${check_src}";
    then
        echo "$src_image does not exist yet, doing nothing"
    else
      echo "copying $src_image to $dest_image"
      copy_image "$src" "$dest_image"
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
    "python-dill:3.10"
    "python-dill:3.10-slim"
    "python-dill:3.11"
    "python-dill:3.11-slim"
    "python-dill:3.12"
    "python-dill:3.12-slim"
    "python-dill:3.13"
    "python-dill:3.13-slim"
    "hail:${HAIL_PIP_VERSION}"
    "hailtop:${HAIL_PIP_VERSION}"
    "vep-grch37-85:${HAIL_PIP_VERSION}"
    "vep-grch38-95:${HAIL_PIP_VERSION}"
)
for image in "${images[@]}"
do
    copy_if_not_present "hailgenetics/${image}" "${DOCKER_PREFIX}/hailgenetics/${image}"
done
