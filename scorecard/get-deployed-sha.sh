#!/bin/bash
set -ex

kubectl get --selector=app=scorecard-deployment deployments -o "jsonpath={.items[*].metadata.labels.hail\.is/sha}"
