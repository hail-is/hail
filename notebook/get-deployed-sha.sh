#!/bin/bash
set -ex

kubectl get --selector=app=notebook deployments -o "jsonpath={.items[*].metadata.labels.hail\.is/sha}"
