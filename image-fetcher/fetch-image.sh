#!/bin/sh

set -ex

gcloud auth activate-service-account \
  --key-file=/secrets/gcr-pull.json

gcloud auth configure-docker

while [ 1 ]
do
    [ $(cat images | wc -l) -eq 0 ] || (cat images | xargs docker pull)
    sleep 10
done
