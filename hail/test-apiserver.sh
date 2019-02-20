#!/bin/bash

set -ex

cleanup() {
    set +e
    trap "" INT TERM
    [[ -z $server_pid ]] || kill -9 $server_pid
}
trap cleanup EXIT
trap "exit 1" INT TERM

# FIXME update pr-builder image
pip install flask

export PYTHONPATH=$(ls $SPARK_HOME/python/lib/py4j-*-src.zip):$SPARK_HOME/python:./python
export JAR=./build/libs/hail-all-spark.jar
export PYSPARK_SUBMIT_ARGS="--conf spark.driver.extraClassPath=./build/libs/hail-all-spark.jar --conf spark.executor.extraClassPath=./build/libs/hail-all-spark.jar pyspark-shell"

python ../apiserver/apiserver/apiserver.py >apiserver.log 2>&1 &
server_pid=$!

../until-with-fuel 30 curl -fL http://localhost:5000/healthcheck

export HAIL_TEST_SERVICE_BACKEND_URL=http://localhost:5000

python -m unittest test.hail.table.test_table.Tests.test_range_table
