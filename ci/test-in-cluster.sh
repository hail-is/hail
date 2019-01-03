#!/bin/bash
set -ex

export UUID=$(./generate-uid.sh)
export SERVICE_NAME=test-ci-$UUID

kubectl expose pod $POD_NAME --name $SERVICE_NAME \
        --type ClusterIP \
        --port 80 \
        --target-port 5000

cleanup() {
    set +e
    trap "" INT TERM
    if [ -n $batch_pid ]
    then
        batch_pgid=$(ps -o pgid= $batch_pid | grep --only-matching '[0-9]*')
        kill -- -$batch_pgid
        kill -9 -$batch_pid
    fi
    kubectl delete service $SERVICE_NAME
}
trap cleanup EXIT
trap "exit 42" INT TERM

mkdir github-tokens
cp /secrets/user* github-tokens
mkdir oauth-token
cp /secrets/oauth-token oauth-token
mkdir gcloud-token
cp /secrets/hail-ci-0-1.key gcloud-token

export IN_CLUSTER=true
export SELF_HOSTNAME=https://ci.hail.is/$SERVICE_NAME
export BATCH_SERVER_URL=http://127.0.0.1:5001

pushd ../batch
BATCH_PORT=5001 make run & batch_pid=$!
popd

../until-with-fuel 30 curl -fL 127.0.0.1:5001/jobs

gcloud auth activate-service-account --key-file=/secrets/hail-ci-0-1.key

./test-locally.sh
