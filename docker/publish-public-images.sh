#!/bin/bash

set -ex

project=$1

make -C ../hail python/hail/hail_pip_version
cp ../hail/python/hail/hail_pip_version hail/hail_pip_version
hail_pip_version=$(cat hail/hail_pip_version)

build_and_push() {
    name=$1
    base=$2

    docker=hailgenetics/$name:$hail_pip_version
    google=gcr.io/$project/$name:$hail_pip_version
    latest=gcr.io/$project/$name:latest

    docker pull $latest || true
    docker build \
           $name/ \
           -f $name/Dockerfile.out \
           -t $docker \
           -t $google \
           -t $latest \
           --cache-from $latest,$base
    docker push $docker
    docker push $google
    docker push $latest
}

python3 ../ci/jinja2_render.py '{"hail_ubuntu_image":{"image":"hail-ubuntu"}}' hail/Dockerfile hail/Dockerfile.out
build_and_push hail hail-ubuntu

python3 ../ci/jinja2_render.py '{"hail_public_image":{"image":"'hailgenetics/hail:$hail_pip_version'"}}' genetics/Dockerfile genetics/Dockerfile.out
build_and_push genetics hailgenetics/hail:${hail_pip_version}

