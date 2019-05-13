#!/bin/bash

set -ex

cleanup() {
    set +e
    trap "" INT TERM
    [[ -z $server_pid ]] || kill -9 $server_pid
    rm -rf bogus-jwt-test-user-token
}
trap cleanup EXIT
trap "exit 1" INT TERM

rm -rf src
mkdir -p src/test
cp -a ../hail/src/test/resources src/test

export JAR=../hail/build/libs/hail-all-spark.jar
export PYSPARK_SUBMIT_ARGS="--conf spark.driver.extraClassPath=$JAR --conf spark.executor.extraClassPath=$JAR pyspark-shell"

HAIL_JWT_SECRET_KEY_FILE=jwt-test-secret-key python3 apiserver/apiserver.py &
server_pid=$!

../until-with-fuel 30 curl -fL http://localhost:5000/healthcheck

export HAIL_APISERVER_URL=http://localhost:5000
export HAIL_TOKEN_FILE="$(pwd)/jwt-test-user-token"

python3 -m unittest test.hail.table.test_table.Tests.test_range_table
python3 -m unittest test.hail.linalg.test_linalg.Tests.test_matrix_ops
python3 -m pytest test

python create_key.py \
       <(echo "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb") \
       < jwt-test-user.json > bogus-jwt-test-user-token
export HAIL_TOKEN_FILE="$(pwd)/bogus-jwt-test-user-token"
python3 <<EOF
import hail as hl
import requests

try:
    hl.utils.range_table(1000)._force_count()
    assert False
except requests.exceptions.HTTPError as err:
    assert err.response.status_code == 401
EOF

unset HAIL_TOKEN_FILE
python3 <<EOF
import hail as hl
import re

try:
    hl.init()
    assert False
except ValueError as ve:
    assert re.search('.*cannot create a client without a token.*',
                     str(ve))
EOF
