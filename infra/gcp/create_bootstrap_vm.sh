#!/bin/bash

gcloud compute instances create bootstrap-vm \
    --image=ubuntu-2004-focal-v20220118 \
    --image-project=ubuntu-os-cloud \
    --machine-type=n1-standard-8 \
    --network=default \
    --boot-disk-size=200GB \
    --service-account=terraform@${PROJECT}.iam.gserviceaccount.com \
    --scopes cloud-platform
