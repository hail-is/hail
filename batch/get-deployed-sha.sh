#!/bin/bash
set -ex

kubectl get --selector=app=batch-deployment deployments -o "jsonpath={.items[*].metadata.labels.hail\.is/sha}"
