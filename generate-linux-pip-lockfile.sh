#!/bin/bash
# Some service dependencies are linux-specific and the lockfile
# would differ when generated on MacOS, so we generate the lockfile
# on a linux image.

set -ex

PIP_COMPILE_IMAGE=hail-pip-compile:latest

if [[ "$(docker images -q $PIP_COMPILE_IMAGE 2>/dev/null)" == "" ]]; then
    docker build -t $PIP_COMPILE_IMAGE -f - . <<EOF
FROM python:3.8-slim
RUN pip install pip-tools==6.13.0
EOF
fi

package=$1
docker run --rm -it \
    -v $HAIL:/hail \
    $PIP_COMPILE_IMAGE \
    pip-compile --upgrade hail/$package/requirements.txt --output-file=hail/$package/pinned-requirements.txt
