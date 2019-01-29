#!/bin/bash
set -ex

SPARK_VERSION=2.2.0
BRANCH=0.2
CLOUDTOOLS_VERSION=3
HASH_TARGET=gs://hail-common/builds/${BRANCH}/latest-hash/cloudtools-${CLOUDTOOLS_VERSION}-spark-${SPARK_VERSION}.txt

gsutil cat ${HASH_TARGET} || true
