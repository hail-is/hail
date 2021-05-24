#!/bin/bash

set -ex

docker_prefix=$1

make -C ../hail python/hail/hail_pip_version
cp ../hail/python/hail/hail_pip_version hail/hail_pip_version
hail_pip_version=$(cat hail/hail_pip_version)

build_and_push() {
    name=$1

    versioned_short=hailgenetics/$name:$hail_pip_version
    versioned_full=$docker_prefix/$versioned_short
    cache=$docker_prefix/hailgenetics/$name:cache

    DOCKER_BUILDKIT=1 docker build \
        --file $name/Dockerfile.out \
        --cache-from ${cache} \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        --tag $versioned_short \
        --tag $versioned_full \
        --tag $cache \
        ${name}

    time DOCKER_BUILDKIT=1 docker push ${versioned_short}
    time DOCKER_BUILDKIT=1 docker push ${versioned_full}
    time DOCKER_BUILDKIT=1 docker push ${cache}
}

python3 ../ci/jinja2_render.py '{"hail_ubuntu_image":{"image":"'$(cat hail-ubuntu-image-ref)'"}}' hail/Dockerfile hail/Dockerfile.out
build_and_push hail

python3 ../ci/jinja2_render.py '{"hail_public_image":{"image":"'hailgenetics/hail:$hail_pip_version'"}}' genetics/Dockerfile genetics/Dockerfile.out
build_and_push genetics
