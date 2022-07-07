#!/bin/bash

set -ex

third_party_images=$(cat third-party/images.txt)
first_party_images=$(cat hailgenetics-images.txt)
hail_pip_version=$(cat ../hail/python/hail/hail_pip_version)

if [ -z "${DOCKER_PREFIX}" ]
then
    echo ERROR: DOCKER_PREFIX must be set before running this
    exit 1
fi

# if command -v skopeo
# then
#     copy_image() {
#         skopeo copy --override-os linux --override-arch amd64 docker://docker.io/$1 docker://$2
#     }
# else
echo Could not find skopeo, falling back to docker which will be slower.
copy_image() {
    docker pull $1
    docker tag $1 $2
    docker push $2
}
# fi


for image in ${third_party_images}
do
    dest="${DOCKER_PREFIX}/${image}"
    copy_image $image $dest
done

for image in ${first_party_images}
do
    dest="${DOCKER_PREFIX}/${image}:${hail_pip_version}"
    copy_image hailgenetics/${image}:${hail_pip_version} $dest
done
