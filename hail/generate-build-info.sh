#!/bin/sh

set -e

REVISION=$(git rev-parse HEAD)
DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ)
BRANCH=$(git rev-parse --abbrev-ref HEAD)
URL=$(git config --get remote.origin.url)
SPARK_VERSION=$1
HAIL_PIP_VERSION=$2

echo_build_properties() {
  echo "[Build Metadata]"
  echo user=$USER
  echo revision=$REVISION
  echo branch=$BRANCH
  echo date=$DATE
  echo url=$URL
  echo sparkVersion=$SPARK_VERSION
  echo hailPipVersion=$HAIL_PIP_VERSION
}

mkdir -p src/main/resources/

echo_build_properties $1 $2 > "src/main/resources/build-info.properties"
printf ${HAIL_PIP_VERSION} > python/hail/hail_pip_version
printf ${HAIL_PIP_VERSION}-${REVISION:0:12} > python/hail/hail_version
