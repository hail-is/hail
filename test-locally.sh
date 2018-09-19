#!/bin/bash
set -ex

export REPO_NAME=ci-test-$(LC_CTYPE=C LC_ALL=C tr -dc 'a-z0-9' < /dev/urandom | head -c 8)
export WATCHED_TARGETS='[["hail-ci-test/'${REPO_NAME}':master", true]]'

set +x
TOKEN=$(cat github-tokens/user1)
set -x

cleanup() {
    set - INT TERM
    set +e
    kill $(cat ci.pid)
    rm -rf ci.pid
    set +x
    curl -XDELETE \
         https://api.github.com/repos/hail-ci-test/${REPO_NAME} \
         -H "Authorization: token ${TOKEN}"
    set -x
}
trap cleanup EXIT

trap "exit 24" INT TERM

# create the temp repo
set +x
curl -XPOST \
     https://api.github.com/orgs/hail-ci-test/repos \
     -H "Authorization: token ${TOKEN}" \
     -d "{ \"name\" : \"${REPO_NAME}\" }"
set -x

# start CI system
source activate hail-ci
python ci/ci.py & echo $! > ci.pid
sleep 10

# upload files to temp repo
# https://unix.stackexchange.com/questions/30091/fix-or-alternative-for-mktemp-in-os-x
REPO_DIR=$(mktemp -d 2>/dev/null || mktemp -d -t 'mytmpdir')
cp test-repo/* ${REPO_DIR}
pushd ${REPO_DIR}
git init
git config user.email 'ci-automated-tests@broadinstitute.org'
git config user.name 'ci-automated-tests'
set +x
git remote add origin \
    https://${TOKEN}@github.com/hail-ci-test/${REPO_NAME}.git
set -x
git add *
git commit -m 'inital commit'
git push origin master:master
popd

# setup webhooks for temp repo
set +x
./setup-endpoints.sh hail-ci-test/${REPO_NAME} ${TOKEN} ${SELF_HOSTNAME}
set -x

export PYTHONPATH=$PYTHONPATH:${PWD}/ci
pytest -vv test/test-ci.py
