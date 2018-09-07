#!/bin/bash

set -ex

PUBLISHED=$(pip --no-cache-dir show cloudtools | grep 'Version:' | sed 's/Version: //')
CURRENT=$(grep 'version=' setup.py | sed -E "s/.*version=\'(.*)\',/\1/")

if [[ "${PUBLISHED}" != "${CURRENT}" ]]
then
    echo deploying ${CURRENT}, was ${PUBLISHED}
    set +x
    export TWINE_USERNAME=$(cat /secrets/pypi-username)
    export TWINE_PASSWORD=$(cat /secrets/pypi-password)
    set -x
    make deploy
else
    echo nothing to do ${PUBLISHED} == ${CURRENT}
fi
