#!/bin/bash
set -ex

gsutil -m cp gs://hail-cseed/cs-hack/run-worker.sh gs://hail-cseed/cs-hack/worker.py /

nohup /bin/bash run-worker.sh >run-worker.log 2>&1 &
