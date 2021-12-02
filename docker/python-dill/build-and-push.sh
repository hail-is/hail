#!/bin/bash

set -ex

for version in 3.6 3.6-slim 3.7 3.7-slim 3.8 3.8-slim
do
    python3 ../../../ci/jinja2_render.py '{"global":{"docker_prefix":'${DOCKER_PREFIX}'}}' python-dill/Dockerfile.$version python-dill/Dockerfile.$version.rendered
    ../../docker-build.sh . Dockerfile.$version.rendered $(DOCKER_PREFIX)/hailgenetics/python-dill:$version
    DOCKER_BUILDKIT=1 docker tag $(DOCKER_PREFIX)/hailgenetics/python-dill:$version docker.io/hailgenetics/python-dill:$version
    time DOCKER_BUILDKIT=1 docker push docker.io/hailgenetics/python-dill:$version
done
