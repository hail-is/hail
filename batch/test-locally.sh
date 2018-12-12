#!/bin/bash
# do not execute this file, use the Makefile

set -ex

PYTEST_ARGS=${PYTEST_ARGS:- -v --failed-first}

cleanup() {
    set +e
    trap "" INT TERM
    [[ -z $server_pid ]] || kill -9 $server_pid
}
trap cleanup EXIT
trap "exit 24" INT TERM

python -c 'import batch.server; batch.server.serve(5000)' &
server_pid=$!

../until-with-fuel 30 curl -fL 127.0.0.1:5000/jobs

POD_IP='127.0.0.1' BATCH_URL='http://127.0.0.1:5000' python -m pytest ${PYTEST_ARGS} test
