#!/bin/sh

set -ex

gcloud auth activate-service-account \
  --key-file=/secrets/gcr-pull.json

gcloud auth configure-docker

while true
do
    for image in `curl -sSL http://notebook/worker-image`
    do
    	docker pull $image
    done
    sleep 360
done
