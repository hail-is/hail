#!/bin/bash
set -ex

ROOT=$(cd .. && pwd)

SPARK_VERSION=2.2.0
BRANCH=devel
CLOUDTOOLS_VERSION=2
HASH_TARGET=gs://hail-common/builds/${BRANCH}/latest-hash/cloudtools-${CLOUDTOOLS_VERSION}-spark-${SPARK_VERSION}.txt
SHA=$(git rev-parse --short=12 HEAD)

DEPLOYED_SHA=$(gsutil cat ${HASH_TARGET})

if [[ $SHA == $DEPLOYED_SHA ]]; then
    exit 0
fi

NEEDS_REDEPLOY=$(cd $ROOT && python3 project-changed.py $DEPLOYED_SHA hail)
if [[ $NEEDS_REDEPLOY = no ]]; then
    exit 0
fi

gcloud auth activate-service-account \
  --key-file=/secrets/ci-deploy-0-1--hail-is-hail.json

source activate hail

# build jar, zip, and distribution
GRADLE_OPTS=-Xmx2048m ./gradlew \
           shadowJar \
           archiveZip \
           makeDocs \
           createPackage \
           --gradle-user-home /gradle-cache

# update jar, zip, and distribution
GS_JAR=gs://hail-common/builds/${BRANCH}/jars/hail-${BRANCH}-${SHA}-Spark-${SPARK_VERSION}.jar
gsutil cp build/libs/hail-all-spark.jar ${GS_JAR}
gsutil acl set public-read ${GS_JAR}

GS_HAIL_ZIP=gs://hail-common/builds/${BRANCH}/python/hail-${BRANCH}-${SHA}.zip
gsutil cp build/distributions/hail-python.zip ${GS_HAIL_ZIP}
gsutil acl set public-read ${GS_HAIL_ZIP}

DISTRIBUTION=gs://hail-common/distributions/${BRANCH}/Hail-${BRANCH}-${SHA}-Spark-${SPARK_VERSION}.zip
gsutil cp build/distributions/hail.zip $DISTRIBUTION
gsutil acl set public-read $DISTRIBUTION

CONFIG=gs://hail-common/builds/${BRANCH}/config/hail-config-${BRANCH}-${SHA}.json
python ./create_config_file.py $BRANCH ./hail-config-${BRANCH}-${SHA}.json
gsutil cp hail-config-${BRANCH}-${SHA}.json ${CONFIG}
gsutil acl set public-read $CONFIG

DOCS=gs://hail-common/builds/${BRANCH}/docs/hail-${BRANCH}-docs-${SHA}.tar.gz
tar cvf hail-${BRANCH}-docs-${SHA}.tar.gz -C build www
gsutil cp hail-${BRANCH}-docs-${SHA}.tar.gz ${DOCS}
gsutil acl set public-read ${DOCS}

echo ${SHA} > latest-hash-spark-${SPARK_VERSION}.txt
gsutil cp ./latest-hash-spark-${SPARK_VERSION}.txt ${HASH_TARGET}
gsutil acl set public-read ${HASH_TARGET}
