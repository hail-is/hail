#!/bin/bash
set -ex

docker info

mkdir /shared

INST_DIR=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/inst_dir")

gsutil cp $INST_DIR/config.json /config.json

python3 /run-task.py >run-task2.log 2>&1
echo $? > run-task2.ec

gsutil cp run-task2.log run-task2.ec $INST_DIR/

# terminate
export NAME=$(curl http://metadata.google.internal/computeMetadata/v1/instance/name -H 'Metadata-Flavor: Google')
export ZONE=$(curl http://metadata.google.internal/computeMetadata/v1/instance/zone -H 'Metadata-Flavor: Google')
gcloud -q compute instances delete $NAME --zone=$ZONE
