#!/bin/sh

set -e

REVISION=$(git rev-parse HEAD)
DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ)
DATE=$(git rev-parse --abbrev-ref HEAD)
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
cat > python/hail/_generated_version_info.py <<EOF
hail_pip_version = "${HAIL_PIP_VERSION}"
hail_version = hail_pip_version + "-${REVISION:0:12}"
EOF
