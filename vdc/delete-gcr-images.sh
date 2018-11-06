#!/bin/bash
set -ex

PROJECT=$(gcloud config get-value project)
echo PROJECT=${PROJECT}

for IMAGE in $(gcloud container images list --format 'get(name)'); do
    echo IMAGE=${IMAGE}
    for DIGEST in $(gcloud container images list-tags ${IMAGE} --format 'get(digest)'); do
        gcloud container images delete -q --force-delete-tags ${IMAGE}@${DIGEST}
    done
done
