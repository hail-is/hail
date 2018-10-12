#!/bin/bash
set -ex

source activate hail-batch

cleanup() {
    set - INT TERM
    set +e
    kill $(cat ci.pid)
    rm -rf ci.pid
}
trap cleanup EXIT

trap "exit 24" INT TERM

# run the server in the background with in-cluster config
python batch/server.py & echo $! > ci.pid

sleep 5

POD_IP='127.0.0.1' BATCH_URL='http://127.0.0.1:5000' python -m unittest test/test_batch.py
