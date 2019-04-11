#!/bin/bash
set -ex

SPARK_VERSION=2.4.0
BRANCH=0.2
CLOUDTOOLS_VERSION=4
HASH_TARGET=gs://hail-common/builds/${BRANCH}/latest-hash/cloudtools-${CLOUDTOOLS_VERSION}-spark-${SPARK_VERSION}.txt

gsutil cat ${HASH_TARGET} || true
