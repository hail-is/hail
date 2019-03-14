#!/bin/bash
# do not execute this file, use the Makefile

set -ex

PYTEST_ARGS=${PYTEST_ARGS:- -v --failed-first}

cleanup() {
    set +e
    trap "" INT TERM
    [[ -z $server_pid ]] || kill -9 $server_pid
    [[ -z $proxy_pid ]] || kill -9 $proxy_pid
}
trap cleanup EXIT
trap "exit 24" INT TERM

if [[ $CLOUD_SQL_PROXY -eq 1 ]]; then
    export CLOUD_SQL_CONFIG_PATH=`pwd`/secrets/batch-test-cloud-sql-config.json
    connection_name=$(jq -r '.connection_name' $CLOUD_SQL_CONFIG_PATH)
    port=$(jq -r '.port' $CLOUD_SQL_CONFIG_PATH)
    ./cloud_sql_proxy -instances=$connection_name=tcp:$port &
    proxy_pid=$!
else
    export CLOUD_SQL_CONFIG_PATH=/secrets/batch-test-cloud-sql-config.json
fi

python3 -c 'import batch.server; batch.server.serve(5000)' &
server_pid=$!

../until-with-fuel 30 curl -fL 127.0.0.1:5000/jobs

POD_IP='127.0.0.1' BATCH_URL='http://127.0.0.1:5000' python3 -m pytest ${PYTEST_ARGS} test
