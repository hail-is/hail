#!/bin/bash
set -ex

kubectl get --selector=app=cronus deployments -o "jsonpath={.items[*].metadata.labels.hail\.is/sha}"
