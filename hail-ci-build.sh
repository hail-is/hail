#!/bin/bash
set -ex

HAIL_CHANGED=$(python project-changed.py target/$TARGET_BRANCH hail)
if [[ $HAIL_CHANGED != no ]]; then
    (cd hail && /bin/bash hail-ci-build.sh)
fi
