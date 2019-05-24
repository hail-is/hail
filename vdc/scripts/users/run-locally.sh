#!/bin/bash
if [ $# -ne 6 ]
  then
    echo "Usage ./run-locally.sh <username> <kube_namespace> <create|delete>"
    exit
fi

cleanup() {
    set +e
    [[ -z $proxy_pid ]] || kill -9 $proxy_pid
    exit
}

trap cleanup EXIT
    instance=$(cat instance.cloudsql)
    ./cloud_sql_proxy -instances="$instance"=tcp:3306 &
    proxy_pid=$!

    BATCH_USE_KUBE_CONFIG=1 SQL_HOST=127.0.0.1 python ./user_data.py "$@"
