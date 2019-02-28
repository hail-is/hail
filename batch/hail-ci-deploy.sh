#!/bin/bash
set -ex

. ../loadconda
conda activate hail-batch

gcloud -q auth activate-service-account \
  --key-file=/secrets/gcr-push-service-account-key.json

gcloud -q auth configure-docker

kubectl delete persistvolumeclaim --all --namespace batch-pods
kubectl delete namespace test
# when we have k8s 1.12 https://github.com/kubernetes/kubernetes/pull/64034:
# kubectl wait --for=delete namespace/test
: $((delay = 1))
while kubectl get namespace test
do
    : $((delay = delay + delay))
    sleep ${delay}
done
kubectl create -f test-namespace.yaml

make deploy
