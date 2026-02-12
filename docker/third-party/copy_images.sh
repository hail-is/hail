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
    # If DOCKERHUB_PREFIX is set, prepend it to the source image
    if [ -n "${DOCKERHUB_PREFIX}" ]
    then
        # Handle both bare images (e.g., "ubuntu") and namespaced images (e.g., "envoyproxy/envoy:v1.33.0")
        # For bare images, we need to add "library/" prefix
        if [[ "$image" == *"/"* ]]
        then
            # Already namespaced (e.g., "envoyproxy/envoy:v1.33.0")
            src="${DOCKERHUB_PREFIX}/${image}"
        else
            # Bare image (e.g., "ubuntu"), need to add "library/" prefix
            src="${DOCKERHUB_PREFIX}/library/${image}"
        fi
    else
        # No dockerhub prefix, use docker.io as before
        src="$image"
    fi
    copy_image "$src" "$dest"
done
