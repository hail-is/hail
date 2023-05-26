#!/bin/bash
# Some service dependencies are linux-specific and the lockfile
# would differ when generated on MacOS, so we generate the lockfile
# on a linux image.

set -ex

source hail-pip-compile.sh

docker run --rm -it \
       -v $HAIL:/hail \
       $PIP_COMPILE_IMAGE \
       /bin/bash -c "cd /hail && bash check_pip_requirements.sh $*"
