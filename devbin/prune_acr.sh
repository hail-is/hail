#!/bin/bash

set -ex

REGISTRY=$1
RESOURCE_GROUP=$2

if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: sh prune_acr.sh <REGISTRY> <RESOURCE_GROUP>"
  exit 1
fi

# Environment variable for container command line
# Notice the --dry-run is on
PURGE_CMD="acr purge \
  --filter 'auth:^cache-pr-.*' \
  --filter 'base:^cache-pr-.*' \
  --filter 'base_spark_3_2:^cache-pr-.*' \
  --filter 'batch:^cache-pr-.*' \
  --filter 'batch-driver-nginx:^cache-pr-.*' \
  --filter 'batch-jvm:^cache-pr-.*' \
  --filter 'batch-worker:^cache-pr-.*' \
  --filter 'benchmark:^cache-pr-.*' \
  --filter 'blog_nginx:^cache-pr-.*' \
  --filter 'ci:^cache-pr-.*' \
  --filter 'ci-hello:^cache-pr-.*' \
  --filter 'ci-intermediate:^cache-pr-.*' \
  --filter 'ci-utils:^cache-pr-.*' \
  --filter 'create_certs_image:^cache-pr-.*' \
  --filter 'echo:^cache-pr-.*' \
  --filter 'grafana:^cache-pr-.*' \
  --filter 'grafana_nginx:^cache-pr-.*' \
  --filter 'hail-base:^cache-pr-.*' \
  --filter 'hail-build:^cache-pr-.*' \
  --filter 'hail-build-spark-32:^cache-pr-.*' \
  --filter 'hail-buildkit:^cache-pr-.*' \
  --filter 'hailgenetics/hail:^cache-pr-.*' \
  --filter 'hail-pip-installed-python36:^cache-pr-.*' \
  --filter 'hail-pip-installed-python37:^cache-pr-.*' \
  --filter 'hail-pip-installed-python38:^cache-pr-.*' \
  --filter 'hail-run:^cache-pr-.*' \
  --filter 'hail-run-tests:^cache-pr-.*' \
  --filter 'hail-ubuntu:^cache-pr-.*' \
  --filter 'memory:^cache-pr-.*' \
  --filter 'monitoring:^cache-pr-.*' \
  --filter 'notebook:^cache-pr-.*' \
  --filter 'notebook_nginx:^cache-pr-.*' \
  --filter 'prom_nginx:^cache-pr-.*' \
  --filter 'prometheus:^cache-pr-.*' \
  --filter 'service-base:^cache-pr-.*' \
  --filter 'service-java-run-base:^cache-pr-.*' \
  --filter 'test-benchmark:^cache-pr-.*' \
  --filter 'test-ci:^cache-pr-.*' \
  --filter 'test-monitoring:^cache-pr-.*' \
  --filter 'test_hello_create_certs_image:^cache-pr-.*' \
  --dry-run \
  --untagged \
  --ago '2d' \
  --keep 3"


# Run for max 8 hour timeout
# Task will run remotely as a task in Azure's Container Registry
az acr run \
  --timeout 28800 \
  --cmd "$PURGE_CMD" \
  --registry $REGISTRY \
  --resource-group $RESOURCE_GROUP \
  /dev/null
