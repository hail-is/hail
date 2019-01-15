#!/bin/bash

set -ex

cd $(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

published=$(
    pip --no-cache-dir search hail \
     | grep '^hail ' \
     | sed 's/hail (//' \
     | sed 's/).*//')
current=$(cat src/hail/hail_pip_version)

if [[ "${published}" != "${current}" ]]
then
    echo deploying ${current}, was ${published}
	  rm -rf python/dist
    python setup.py sdist bdist_wheel
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

    git tag ${current} -m "Hail version ${current}"
    git push https://github.com/hail-is/hail.git ${current}
else
    echo nothing to do ${published} == ${current}
fi
