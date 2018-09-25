#!/bin/bash
set -ex

gcloud -q auth activate-service-account \
  --key-file=/secrets/gcr-push-service-account-key.json

gcloud -q auth configure-docker

make push-site deploy-site
