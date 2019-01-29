#!/bin/bash
set -ex

. ../loadconda
conda activate hail-batch

gcloud -q auth activate-service-account \
  --key-file=/secrets/gcr-push-service-account-key.json

gcloud -q auth configure-docker

make deploy
