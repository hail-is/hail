#!/bin/bash

set -ex

docker_prefix=$1

make -C ../hail python/hail/hail_pip_version
cp ../hail/python/hail/hail_pip_version hail/hail_pip_version
hail_pip_version=$(cat hail/hail_pip_version)

build_and_push() {
    name=$1
    base=$2

    versioned_short=hailgenetics/$name:$hail_pip_version
    versioned_full=$docker_prefix/$versioned_short
    latest_full=$docker_prefix/hailgenetics/$name:latest

    docker pull $latest || true
    docker build \
           $name/ \
           -f $name/Dockerfile.out \
           -t $versioned_short \
           -t $versioned_full \
           -t $latest_full \
           --cache-from $latest_full,$base
    docker push $versioned_short
    docker push $versioned_full
    docker push $latest_full
}

python3 ../ci/jinja2_render.py '{"hail_ubuntu_image":{"image":"hail-ubuntu"}}' hail/Dockerfile hail/Dockerfile.out
build_and_push hail hail-ubuntu

python3 ../ci/jinja2_render.py '{"hail_public_image":{"image":"'hailgenetics/hail:$hail_pip_version'"}}' genetics/Dockerfile genetics/Dockerfile.out
build_and_push genetics hailgenetics/hail:${hail_pip_version}
