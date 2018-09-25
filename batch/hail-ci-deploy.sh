#!/bin/bash
set -ex

source activate hail-batch

gcloud -q auth activate-service-account \
  --key-file=/secrets/gcr-push-service-account-key.json

gcloud -q auth configure-docker

# requires docker
make push-batch

SHA=$(git rev-parse --short=12 HEAD)

sed -e "s,@sha@,$SHA," \
    -e "s,@image@,$(cat batch-image)," \
    < deployment.yaml.in > deployment.yaml

kubectl apply -f deployment.yaml
kubectl rollout status deployment batch-deployment

# ci can't recover from batch restart yet
kubectl delete pods -l app=hail-ci
