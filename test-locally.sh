#!/bin/bash
set -ex

function cleanup {
    kill $(cat ci.pid)
    rm -rf ci.pid
}
trap cleanup EXIT

export WATCHED_TARGETS='[["hail-is/ci-test:master", true]]'

source activate hail-ci
python ci/ci.py & echo $! > ci.pid
sleep 5
PYTHONPATH=$PYTHONPATH:${PWD}/ci pytest -vv test/test-ci.py

