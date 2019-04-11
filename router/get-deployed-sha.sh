#!/bin/bash
set -ex

kubectl -n default get --selector=app=router deployments -o "jsonpath={.items[*].metadata.labels.hail\.is/sha}"
