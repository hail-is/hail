#!/bin/bash
set -ex

function cleanup {
    kill $(cat ci.pid)
    rm -rf ci.pid
}
trap cleanup EXIT

export WATCHED_REPOS='["hail-is/ci-test"]'

source activate hail-ci
python ci/ci.py & echo $! > ci.pid
sleep 5
python -m unittest ci/test-ci2.py ${UNITTEST_ARGS}

