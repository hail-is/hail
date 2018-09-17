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
    trap - INT TERM
    kubectl delete service $SERVICE_NAME
}
trap cleanup EXIT

trap "exit 42" INT TERM

get_ip() {
    kubectl get service $SERVICE_NAME --no-headers | awk '{print $4}'
}

cp /secrets/user* github-tokens
mkdir oauth-token
cp /secrets/oauth-token oauth-token
mkdir gcloud-token
cp /secrets/hail-ci-0-1.key gcloud-token

export IN_CLUSTER=true
export SELF_HOSTNAME=https://ci.hail.is/$SERVICE_NAME
export BATCH_SERVER_URL=http://batch

./test-locally.sh
