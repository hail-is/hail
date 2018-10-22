#!/bin/bash

set -ex

conda env create -f environment.yml || conda env update -f environment.yml
make build
SVC_ACCT_NAME=$(whoami)-gke
KEY_FILE=~/.hail/gke/svc-acct/${SVC_ACCT_NAME}.json
if [ ! -e "${KEY_FILE}" ]
then
    gcloud iam service-accounts create ${SVC_ACCT_NAME}
    gcloud projects add-iam-policy-binding \
           broad-ctsa \
           --member "serviceAccount:${SVC_ACCT_NAME}@broad-ctsa.iam.gserviceaccount.com" \
           --role "roles/owner"
    mkdir -p $(dirname ${KEY_FILE})
    gcloud iam service-accounts keys create \
           ${KEY_FILE} \
           --iam-account ${SVC_ACCT_NAME}@broad-ctsa.iam.gserviceaccount.com
fi

export GOOGLE_APPLICATION_CREDENTIALS="${KEY_FILE}"
