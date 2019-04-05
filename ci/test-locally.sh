#!/bin/bash
set -ex

python3 -m pip install -U ../batch

export UUID=${UUID:-$(../generate-uid.sh)}
export REPO_NAME=ci-test-$UUID
export WATCHED_TARGETS='[["hail-ci-test/'${REPO_NAME}':master", true]]'

set +x
TOKEN=$(cat github-tokens/user1)
set -x

cleanup() {
    set "" INT TERM
    set +e
    kill $(cat ci.pid)
    rm -rf ci.pid
    set +x
    curl -XDELETE \
         -i \
         https://api.github.com/repos/hail-ci-test/${REPO_NAME} \
         -H "Authorization: token ${TOKEN}"
    set -x
}
trap cleanup EXIT

trap "exit 24" INT TERM

# create the temp repo
set +x
curl -XPOST \
     -i \
     https://api.github.com/orgs/hail-ci-test/repos \
     -H "Authorization: token ${TOKEN}" \
     -d "{ \"name\" : \"${REPO_NAME}\" }"
set -x

# wait for create to propagate
# FIXME poll?
sleep 5

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

# start CI system
python3 run_ci.py --debug & echo $! > ci.pid

sleep 10

# setup webhooks for temp repo
set +x
./setup-endpoints.sh hail-ci-test/${REPO_NAME} ${TOKEN} ${SELF_HOSTNAME}
set -x

PYTHONPATH=${PWD}:${PYTHONPATH} python3 -m pytest -vv --failed-first --maxfail=1 test/test-ci.py
