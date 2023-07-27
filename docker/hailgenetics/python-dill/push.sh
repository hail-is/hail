#!/bin/bash
cd $(dirname $0)
set -ex

for version in 3.8 3.8-slim 3.9 3.9-slim 3.10 3.10-slim 3.11 3.11-slim
do
    public=hailgenetics/python-dill:$version

    DOCKER_BUILDKIT=1 docker build \
        --platform linux/amd64 \
        --build-arg PYTHON_VERSION=$version \
        --file Dockerfile \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        --tag ${public} \
        .

    time DOCKER_BUILDKIT=1 docker push ${public}
done
