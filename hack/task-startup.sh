#!/bin/bash
set -ex

gsutil -m cp gs://hail-cseed/cs-hack/run-task.sh gs://hail-cseed/cs-hack/run-task.py /

chmod +x /run-task.sh

nohup /run-task.sh >/run-task1.out 2>/run-task1.err &
