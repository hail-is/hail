#!/bin/bash

set -ex

reqs=$1
pinned_reqs=$2

# Some service dependencies are linux-specific and the lockfile
# would differ when generated on MacOS, so we generate the lockfile
# on a linux image.
if [[ "$(uname)" == 'Linux' ]]; then
    pip-compile --upgrade $reqs --output-file=$pinned_reqs
else
	docker run --rm -it \
        -v ${HAIL_HAIL_DIR}:/hail \
		python:3.7-slim \
		/bin/bash -c "pip install pip-tools && cd /hail && pip-compile --upgrade $reqs --output-file=$pinned_reqs"
fi
