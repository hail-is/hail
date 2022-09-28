#!/bin/bash

set -ex

source ../copy_image.sh

images=$(cat images.txt)

if [ -z "${DOCKER_PREFIX}" ]
then
    echo ERROR: DOCKER_PREFIX must be set before running this
    exit 1
fi

for image in ${images}
do
    dest="${DOCKER_PREFIX}/${image}"
    copy_image $image $dest
done
