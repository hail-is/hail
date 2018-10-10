#!/bin/bash
set -ex

kubectl get --selector=app=site-deployment deployments -o "jsonpath={.items[*].metadata.labels.hail\.is/sha}"
