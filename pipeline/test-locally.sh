#!/bin/bash

set -ex

. activate hail-pipeline

PYTEST_ARGS=${PYTEST_ARGS:- -v --failed-first}

cleanup() {
    set +e
    trap "" INT TERM
    [[ -z $server_pid ]] || kill -9 $server_pid
    [[ -z $proxy_pid ]] || kill -9 $proxy_pid
    [[ -z $last_service_account ]] || gcloud config set account $last_service_account
}

trap cleanup EXIT
trap "exit 24" INT TERM

if [[ $IN_CLUSTER -eq 0 ]]; then
    export CLOUD_SQL_CONFIG_PATH=`pwd`/batch-secrets/batch-test-cloud-sql-config.json
    connection_name=$(jq -r '.connection_name' $CLOUD_SQL_CONFIG_PATH)
    port=$(jq -r '.port' $CLOUD_SQL_CONFIG_PATH)
    ./cloud_sql_proxy -instances=$connection_name=tcp:$port &
    proxy_pid=$!
else
    export CLOUD_SQL_CONFIG_PATH=/batch-secrets/batch-test-cloud-sql-config.json
    pipeline_test_secret_key=/pipeline-secrets/pipeline-test-0-1--hail-is.key
    last_service_account=$(gcloud auth list --filter=status:ACTIVE --format="value(account)")
    gcloud auth activate-service-account pipeline-test-0-1--hail-is@hail-vdc.iam.gserviceaccount.com --key-file $pipeline_test_secret_key 
fi

cd ../batch/
python3 -c 'import batch.server; batch.server.serve()' &
server_pid=$!
cd ../pipeline/

: $((tries = 0))
until curl -fL 127.0.0.1:5000/jobs >/dev/null 2>&1
do
    : $((tries = tries + 1)) && [ $tries -lt 30 ]
    sleep 1
done

BATCH_URL='http://127.0.0.1:5000' pytest ${PYTEST_ARGS} test
