#!/bin/bash
set -ex

PROJECTS='hail batch ci site scorecard cloudtools upload notebook'
SHA=$(git rev-parse --short=12 HEAD)

for project in $PROJECTS; do
    if [[ -e $project/hail-ci-deploy.sh ]]; then
        CHANGED=yes # default
        if [[ -e $project/get-deployed-sha.sh ]]; then
            DEPLOYED_SHA=$(cd $project && /bin/bash get-deployed-sha.sh 2>/dev/null || true)
            if [[ $DEPLOYED_SHA != $SHA ]]; then
                if [[ $(git cat-file -t "$DEPLOYED_SHA" 2>/dev/null || true) == commit ]]; then
                    CHANGED=$(python3 project-changed.py $DEPLOYED_SHA hail)
                fi
            else
                CHANGED=no
            fi
        fi
        
        if [[ $CHANGED != no ]]; then
            (cd $project && /bin/bash hail-ci-deploy.sh)
        fi
    fi
done
