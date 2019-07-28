#!/bin/bash
set -x

export CORES=$(nproc)
export DRIVER=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/driver")
export INST_TOKEN=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/inst_token")

python3 /worker.py

gsutil -m cp worker.log run-worker.log gs://hail-cseed/cs-hack/logs/$INST_TOKEN/

export NAME=$(curl http://metadata.google.internal/computeMetadata/v1/instance/name -H 'Metadata-Flavor: Google')
export ZONE=$(curl http://metadata.google.internal/computeMetadata/v1/instance/zone -H 'Metadata-Flavor: Google')
gcloud -q compute instances delete $NAME --zone=$ZONE
