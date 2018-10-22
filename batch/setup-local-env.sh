#!/bin/bash
# USAGE: source setup-local-env.sh

set -ex

# dependencies
docker version || brew cask install docker # FIXME: non OS X environments
conda -V || (cd /tmp && curl -LO https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh && bash /tmp/Miniconda3-latest-MacOSX-x86_64.sh)
kubectl version || brew install kubernetes-cli # FIXME: non OS X environments

conda env create -f environment.yml || conda env update -f environment.yml
. activate hail-batch
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
