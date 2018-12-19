#!/bin/bash
set -ex

kubectl -n default get --selector=app=hail-ci deployments -o "jsonpath={.items[*].metadata.labels.hail\.is/sha}"
