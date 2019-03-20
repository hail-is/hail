#!/bin/bash
set -ex

export UUID=$(./generate-uid.sh)
export SERVICE_NAME=test-ci-$UUID
export HAIL_TEST_MODE=1
export SELF_HOSTNAME=https://ci.hail.is/$SERVICE_NAME
export BATCH_SERVER_URL=http://127.0.0.1:5001
export REPO_NAME=ci-test-$UUID
export WATCHED_TARGETS='[["hail-ci-test/'${REPO_NAME}':master", true]]'

kubectl expose pod $POD_NAME --name $SERVICE_NAME \
        --type ClusterIP \
        --port 80 \
        --target-port 5000

cleanup() {
    set +e
    trap "" INT TERM
    kubectl delete service $SERVICE_NAME
    if [ -n ${TOKEN} ]
    then
       set +x
       curl -XDELETE \
            -i \
            https://api.github.com/repos/hail-ci-test/${REPO_NAME} \
            -H "Authorization: token ${TOKEN}"
       set -x
    fi
    kill $batch_pid
    kill -9 $batch_pid
    kill $ci_pid
    kill -9 $ci_pid
}
trap cleanup EXIT
trap "exit 42" INT TERM

mkdir github-tokens
cp /secrets/user* github-tokens
mkdir oauth-token
cp /secrets/oauth-token oauth-token
mkdir gcloud-token
cp /secrets/hail-ci-0-1.key gcloud-token

pushd ../batch
conda activate hail-batch
python -c 'import batch.server; batch.server.serve(5001)' & batch_pid=$!
conda deactivate
popd

../until-with-fuel 30 curl -fL 127.0.0.1:5001/alive

gcloud auth activate-service-account --key-file=/secrets/hail-ci-0-1.key

conda activate hail-ci

pip install -U ../batch

set +x
TOKEN=$(cat github-tokens/user1)
set -x

set +x
curl -XPOST \
     -i \
     https://api.github.com/orgs/hail-ci-test/repos \
     -H "Authorization: token ${TOKEN}" \
     -d "{ \"name\" : \"${REPO_NAME}\" }"
set -x

../until-with-fuel 30 curl -fL http://github.com/hail-ci-test/$REPO_NAME

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

python run_ci.py --debug & ci_pid=$!

../until-with-fuel 30 curl -fL localhost:5000/alive

set +x
./setup-endpoints.sh hail-ci-test/${REPO_NAME} ${TOKEN} ${SELF_HOSTNAME}
set -x

pytest ${PYTEST_ARGS: -v --failed-first} test

