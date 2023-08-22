#!/bin/bash
# Some service dependencies are linux-specific and the lockfile
# would differ when generated on MacOS, so we generate the lockfile
# on a linux image.

set -ex

source hail-pip-compile.sh

package=$1
docker run --rm \
    -v $HAIL:/hail \
    $PIP_COMPILE_IMAGE \
    pip-compile --upgrade hail/$package/requirements.txt --output-file=hail/$package/pinned-requirements.txt
