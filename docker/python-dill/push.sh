#!/bin/bash

for version in 3.6 3.6-slim 3.7 3.7-slim 3.8 3.8-slim
do
    sed "s/@PYTHON_VERSION@/$version/g" Dockerfile > Dockerfile.out
    docker build --tag hailgenetics/python-dill:$version - <Dockerfile.out
    docker push hailgenetics/python-dill:$version
    docker tag hailgenetics/python-dill:$version ${DOCKER_PREFIX}/python-dill:$version
    docker push ${DOCKER_PREFIX}/python-dill:$version
done
