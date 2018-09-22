#!/bin/bash
set -ex

HAIL_NEEDS_REDEPLOY=$(python needs-redeploy.py target/$TARGET_BRANCH)
if [[ $HAIL_NEEDS_REDEPLOY != no ]]; then
    (cd hail && /bin/bash hail-ci-build.sh)
fi
