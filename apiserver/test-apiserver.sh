#!/bin/bash

set -ex

cleanup() {
    set +e
    trap "" INT TERM
    [[ -z $server_pid ]] || kill -9 $server_pid
}
trap cleanup EXIT
trap "exit 1" INT TERM

rm -rf src
mkdir -p src/test
cp -a ../hail/src/test/resources src/test

export PYTHONPATH=$(ls $SPARK_HOME/python/lib/py4j-*-src.zip):$SPARK_HOME/python:../hail/python
export JAR=../hail/build/libs/hail-all-spark.jar
export PYSPARK_SUBMIT_ARGS="--conf spark.driver.extraClassPath=$JAR --conf spark.executor.extraClassPath=$JAR pyspark-shell"

python3 apiserver/apiserver.py &
server_pid=$!

../until-with-fuel 30 curl -fL http://localhost:5000/healthcheck

export HAIL_TEST_SERVICE_BACKEND_URL=http://localhost:5000

python3 -m unittest test.hail.table.test_table.Tests.test_range_table
python3 -m unittest test.hail.linalg.test_linalg.Tests.test_matrix_ops
