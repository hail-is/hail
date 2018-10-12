#!/bin/bash
set -ex

kubectl get --selector=app=upload deployments -o "jsonpath={.items[*].metadata.labels.hail\.is/sha}"
