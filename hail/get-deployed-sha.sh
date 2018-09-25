#!/bin/bash
set -ex

SPARK_VERSION=2.2.0
BRANCH=devel
CLOUDTOOLS_VERSION=2
HASH_TARGET=gs://hail-common/builds/${BRANCH}/latest-hash/cloudtools-${CLOUDTOOLS_VERSION}-spark-${SPARK_VERSION}.txt

gsutil cat ${HASH_TARGET} || true
