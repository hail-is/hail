#!/bin/bash

set -ex

. activate hail-pipeline

PYTEST_ARGS=${PYTEST_ARGS:- -v --failed-first}

cleanup() {
    set +e
    trap "" INT TERM
    [[ -z $server_pid ]] || kill -9 $server_pid
}
trap cleanup EXIT
trap "exit 24" INT TERM

cd ../batch/
python -c 'import batch.server; batch.server.serve()' &
server_pid=$!
cd ../pipeline/

: $((tries = 0))
until curl -fL 127.0.0.1:5000/jobs >/dev/null 2>&1
do
    : $((tries = tries + 1)) && [ $tries -lt 30 ]
    sleep 1
done

BATCH_URL='http://127.0.0.1:5000' pytest ${PYTEST_ARGS} test
