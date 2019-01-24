#!/bin/bash

set -ex

cd $(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

cleanup() {
    trap "" INT TERM
    rm hail/hail-all-spark.jar
    rm README.md
    rm -rf build/lib
}
trap cleanup EXIT
trap "exit 24" INT TERM

python3=${HAIL_PYTHON3:-python3}

cp ../build/libs/hail-all-spark.jar hail/
cp ../../README.md .
rm -f dist/*
$python3 setup.py sdist bdist_wheel
ls dist
pip install -U dist/hail-$(cat hail/hail_pip_version)-py3-none-any.whl
