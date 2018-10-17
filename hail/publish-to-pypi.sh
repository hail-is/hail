#!/bin/bash

set -ex

CURRENT=$(cat python/hail/version)

PYPI_VERSIONS=$(python list_pypi_versions.py hail)

set +e
grep -qe "^${CURRENT}$" <(echo "$PYPI_VERSIONS")
ALREADY_PUBLISHED=$?
set -e

if [[ $ALREADY_PUBLISHED -ne 0 ]]
then
    echo deploying ${CURRENT}
    ./gradlew shadowJar
    cp build/libs/hail-all-spark.jar python/hail/hail-all-spark.jar
    cp ../README.md python/
    set +x
    export TWINE_USERNAME=$(cat secrets/pypi-username)
    export TWINE_PASSWORD=$(cat secrets/pypi-password)
    set -x
    cd python
    rm -rf dist
    python setup.py sdist bdist_wheel
    twine upload dist/*
else
    echo nothing to do ${CURRENT} already published
fi
