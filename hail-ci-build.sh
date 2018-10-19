#!/bin/bash
set -ex

PROJECTS='hail batch ci site scorecard cloudtools upload notebook'

for project in $PROJECTS; do
    if [[ -e $project/hail-ci-build.sh ]]; then
        CHANGED=$(python3 project-changed.py target/$TARGET_BRANCH $project)
        if [[ $CHANGED != no ]]; then
            (cd $project && /bin/bash hail-ci-build.sh)
        fi
    fi
done
