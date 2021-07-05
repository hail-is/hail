#!/bin/bash

for version in 3.6 3.6-slim 3.7 3.7-slim 3.8 3.8-slim
do
    sed "s/@PYTHON_VERSION@/$version/g" Dockerfile > Dockerfile.out

    public=hailgenetics/python-dill:$version
    private=${DOCKER_PREFIX}/python-dill:$version
    cache=${DOCKER_PREFIX}/python-dill:cache

    DOCKER_BUILDKIT=1 docker build \
        --file Dockerfile.out \
        --cache-from ${cache} \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        --tag ${public} \
        --tag ${private} \
        --tag ${cache} \
        .

    time DOCKER_BUILDKIT=1 docker push ${public}
    time DOCKER_BUILDKIT=1 docker push ${private}
    time DOCKER_BUILDKIT=1 docker push ${cache}
done
