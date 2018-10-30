#!/bin/sh

set -ex

gcloud auth activate-service-account \
  --key-file=/secrets/gcr-pull.json

gcloud auth configure-docker

while true
do
    curl -sSL http://notebook/worker-image | xargs docker pull
    sleep 360
done
