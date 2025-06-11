#!/bin/bash
# Some service dependencies are linux-specific and the lockfile
# would differ when generated on MacOS, so we generate the lockfile
# on a linux image.

set -ex

package=$1
uv pip compile \
    --python-version 3.9.22 \
    --python-platform linux \
    --upgrade $package/requirements.txt --output-file=$package/pinned-requirements.txt
