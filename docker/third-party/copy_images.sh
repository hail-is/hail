#!/bin/bash

set -ex

images=$(cat images.txt)

if [ -z "${PROJECT}" ]
then
    echo ERROR: PROJECT must be set before running this
    exit 1
fi

for image in ${images}
do
    dest="gcr.io/${PROJECT}/${image}"
    docker pull $image
    docker tag $image $dest
    docker push $dest
done
