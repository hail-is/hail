#!/bin/bash

set -ex

PROJECT=$(gcloud config get-value project)
gcloud compute instances create bootstrap-vm \
  --image-project=ubuntu-os-cloud \
  --image-family=ubuntu-2004-lts \
  --machine-type=n1-standard-8 \
  --boot-disk-size=200GB \
  --service-account=terraform@$PROJECT.iam.gserviceaccount.com \
  --scopes=cloud-platform
