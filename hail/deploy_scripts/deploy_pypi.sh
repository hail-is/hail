#!/bin/bash

set -ex

cd $(CDPATH= cd -- "$(dirname -- "$0")"/.. && pwd)

cleanup() {
    trap "" INT TERM
    rm -rf build/lib
}
trap cleanup EXIT
trap "exit 24" INT TERM

python3=${HAIL_PYTHON3:-python3}

published=$(
    pip --no-cache-dir search hail \
     | grep '^hail ' \
     | sed 's/hail (//' \
     | sed 's/).*//')
current=$(cat python/hail/hail_pip_version)

if [[ "${published}" != "${current}" ]]
then
    echo deploying ${current}, was ${published}
    rm -rf build/deploy_pypi
    mkdir -p build/deploy_pypi
    mkdir -p build/deploy_pypi/src

    cp ../README.md build/deploy_pypi/
    cp deploy_scripts/setup.py build/deploy_pypi/
    cp deploy_scripts/setup.cfg build/deploy_pypi/

    rsync -rv \
      --exclude '__pycache__/' \
      --exclude 'docs/' \
      --exclude '*.log' \
      python/hail build/deploy_pypi/src/

    rsync -rv \
      --exclude '__pycache__/' \
      --exclude '*.log' \
      ../hailctl/python/hailctl build/deploy_pypi/src/

    cp build/libs/hail-all-spark.jar build/deploy_pypi/src/hail/

    cd build/deploy_pypi
    cp src/hail/hail_pip_version src/hailctl/hail_pip_version
    cp src/hail/hail_version src/hailctl/hail_version
    ${python3} setup.py sdist bdist_wheel

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
