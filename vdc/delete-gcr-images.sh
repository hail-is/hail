#!/bin/bash
set -ex

PROJECT=$(gcloud config get-value project)
echo PROJECT=${PROJECT}

for BASE in batch run-letsencrypt scorecard site upload; do
    IMAGE=gcr.io/${PROJECT}/${BASE}
    for DIGEST in $(gcloud container images list-tags ${IMAGE} --format 'get(digest)'); do
        gcloud container images delete -q --force-delete-tags ${IMAGE}@${DIGEST}
    done
done
