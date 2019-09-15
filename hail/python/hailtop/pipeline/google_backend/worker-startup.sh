#!/bin/bash
set -ex

for i in $(seq 1 30); do
    gsutil -m cp gs://hail-common/dev2/pipeline/run-worker.sh gs://hail-common/dev2/pipeline/worker.py /
    if [[ $? = 0 ]]; then
        break;
    fi
    sleep 1
done

nohup /bin/bash run-worker.sh >run-worker.log 2>&1 &
