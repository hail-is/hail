#!/bin/bash

set -ex

for version in 3.8 3.8-slim 3.9 3.9-slim 3.10 3.10-slim
do
    public=hailgenetics/python-dill:$version

    DOCKER_BUILDKIT=1 docker build \
        --build-arg PYTHON_VERSION=$version \
        --file Dockerfile.out \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        --tag ${public} \
        .

    time DOCKER_BUILDKIT=1 docker push ${public}
done
