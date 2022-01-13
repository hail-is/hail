#!/bin/bash

set -ex

images=$(cat images.txt)

if [ -z "${DOCKER_PREFIX}" ]
then
    echo ERROR: DOCKER_PREFIX must be set before running this
    exit 1
fi

if command -v skopeo
then
    copy_image() {
        skopeo copy --override-os linux --override-arch amd64 docker://docker.io/$1 docker://$2
    }
else
    echo Could not find skopeo, falling back to docker which will be slower.
    copy_image() {
        docker pull $1
        docker tag $1 $2
        docker push $2
    }
fi


for image in ${images}
do
    dest="${DOCKER_PREFIX}/${image}"
    copy_image $image $dest
done
