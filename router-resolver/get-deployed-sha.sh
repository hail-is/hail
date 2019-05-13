#!/bin/bash
set -ex

kubectl -n default get --selector=app=router-resolver deployments -o "jsonpath={.items[*].metadata.labels.hail\.is/sha}"
