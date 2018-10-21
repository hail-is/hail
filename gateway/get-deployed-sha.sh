#!/bin/bash
set -ex

kubectl get --selector=app=gateway deployments -o "jsonpath={.items[*].metadata.labels.hail\.is/sha}"
