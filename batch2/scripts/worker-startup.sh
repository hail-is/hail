#!/bin/bash
set -ex

# gsutil -m cp gs://hail-common/dev2/batch2/run-worker.sh /

export BATCH_IMAGE=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/batch_image")

export HOME=/root
nohup /bin/bash docker run -v /var/run/docker.sock:/var/run/docker.sock -p 5000:5000 -d --entrypoint "/bin/bash" $BATCH_IMAGE -c "./run-worker.sh"
# docker run -v /var/run/docker.sock:/var/run/docker.sock -p 5000:5000 -d --entrypoint "/bin/bash" $BATCH_IMAGE -c "python3 -u -m 'batch.agent'"

# nohup /bin/bash run-worker.sh >run-worker.log 2>&1 &
