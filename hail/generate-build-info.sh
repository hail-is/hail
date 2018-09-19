#!/bin/sh

set -e

REVISION=$(git rev-parse HEAD)
DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ)
DATE=$(git rev-parse --abbrev-ref HEAD)
URL=$(git config --get remote.origin.url)
SPARK_VERSION=$1
HAIL_VERSION=$2

echo_build_properties() {
  echo "[Build Metadata]"
  echo user=$USER
  echo revision=$REVISION
  echo branch=$BRANCH
  echo date=$DATE
  echo url=$URL
  echo sparkVersion=$SPARK_VERSION
  echo hailVersion=$HAIL_VERSION
}

mkdir -p src/main/resources/

echo_build_properties $1 $2 > "src/main/resources/build-info.properties"
echo "hail_version = \"${HAIL_VERSION}-${REVISION:0:12}\"" > python/hail/_generated_version_info.py
