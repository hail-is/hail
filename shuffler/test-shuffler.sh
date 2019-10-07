#!/bin/sh

set -ex

mkdir -p build
rm -rf build/shuffler.stdout.stderr
sbt run > build/shuffler.stdout.stderr 2>&1 & PID=$!

cleanup() {
    set +e
    trap "" INT TERM
    kill $PID
    kill -9 $PID
    if [[ ${TEST_EC} != 0 ]]
    then
       echo ">>>> sbt run stdout & stderr <<<<"
       cat build/shuffler.stdout.stderr
    fi
}
trap cleanup EXIT
trap "exit 24" INT TERM

../until-with-fuel 5 curl -fL localhost:5000/healthcheck

${HAIL_PYTHON3} -m pytest test \
		-v \
		-n ${PARALLELISM} \
		--dist=loadscope \
    --instafail \
		--noconftest \
		--color=no \
		-r A \
		--html=build/reports/pytest.html \
		--self-contained-html \
		${PYTEST_ARGS}
echo bye
TEST_EC=$?
