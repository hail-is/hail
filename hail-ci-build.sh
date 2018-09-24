#!/bin/bash
set -ex

PROJECTS='hail batch ci site scorecard cloudtools'

for project in $PROJECTS; do
    CHANGED=$(python3 project-changed.py target/$TARGET_BRANCH $project)
    if [[ $CHANGED != no ]]; then
	if [[ -e $project/hail-ci-build.sh ]]; then
	    (cd $project && /bin/bash hail-ci-build.sh)
	fi
    fi
fi
