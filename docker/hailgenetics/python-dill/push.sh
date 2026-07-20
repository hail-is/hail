#!/bin/bash
cd $(dirname $0)
set -ex

for version in 3.12 3.12-slim 3.13 3.13-slim 3.14 3.14-slim
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
