#!/bin/bash

set -ex

reqs=$1
pinned_reqs=$2

# Some service dependencies are linux-specific and the lockfile
# would differ when generated on MacOS, so we generate the lockfile
# on a linux image.
if [[ "$(uname)" == 'Linux' ]]; then
    # `pip install pip-tools` on dataproc by default installs into the
    # user's local bin which is not on the PATH
    PATH="$PATH:$HOME/.local/bin" pip-compile --upgrade $reqs --output-file=$pinned_reqs
else
	docker run --rm \
        -v ${HAIL_HAIL_DIR}:/hail \
		python:3.9-slim \
		/bin/bash -c "pip install 'pip-tools==6.13.0' && cd /hail && pip-compile --upgrade $reqs --output-file=$pinned_reqs"
fi
