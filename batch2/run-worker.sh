#!/bin/bash
set -x

export CORES=$(nproc)
export NAMESPACE=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/namespace")
export BATCH_WORKER_IMAGE=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/batch_worker_image")
export INST_TOKEN=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/inst_token")
export LOG_ROOT=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/log_root")
export INTERNAL_IP=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/ip")
export NAME=$(curl http://metadata.google.internal/computeMetadata/v1/instance/name -H 'Metadata-Flavor: Google')
export ZONE=$(curl http://metadata.google.internal/computeMetadata/v1/instance/zone -H 'Metadata-Flavor: Google')

export WORKER_LOG_FILE=$LOG_ROOT/worker/$NAME/worker.log

python3 -u -m "batch.worker" > worker.log 2>&1

gsutil -m cp worker.log $WORKER_LOG_FILE

gcloud -q compute instances delete $NAME --zone=$ZONE
