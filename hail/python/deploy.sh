#!/bin/bash

set -ex

cd $(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
python3=${HAIL_PYTHON3:-python3}

published=$(
    pip --no-cache-dir search hail \
     | grep '^hail ' \
     | sed 's/hail (//' \
     | sed 's/).*//')
current=$(
    grep 'hail_pip_version = ' hail/_generated_version_info.py \
     | sed -E "s/.*hail_pip_version = \"(.*)\"/\1/")

if [[ "${published}" != "${current}" ]]
then
    echo deploying ${current}, was ${published}
    cp ../build/libs/hail-all-spark.jar hail/
    rm -f dist/*
    $python3 setup.py sdist bdist_wheel
    if [[ -e /secrets/pypi-username && -e /secrets/pypi-password ]]
    then
        set +x
        TWINE_USERNAME=$(cat /secrets/pypi-username) \
          TWINE_PASSWORD=$(cat /secrets/pypi-password) \
          twine upload dist/*
        set -x
    else
        set +x
        TWINE_USERNAME=$(cat $HAIL_TWINE_CREDS_FOLDER/pypi-username) \
          TWINE_PASSWORD=$(cat $HAIL_TWINE_CREDS_FOLDER/pypi-password) \
          twine upload dist/*
        set -x
    fi
else
    echo nothing to do ${published} == ${current}
fi
