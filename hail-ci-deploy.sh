set -ex

SPARK_VERSION=2.0.2
BRANCH_TARGET_NAME=0.1
HASH_TARGET=gs://hail-common/builds/${BRANCH_TARGET_NAME}/latest-hash-spark-${SPARK_VERSION}.txt
SHA=$(git rev-parse --short=12 HEAD)

if [[ "${SHA}" == "$(gsutil cat ${HASH_TARGET})" ]]
then
    exit 0
fi

gcloud auth activate-service-account \
  --key-file=/secrets/ci-deploy-0-1--hail-is-hail.json

source activate hail-0.1-dev

# build jar, zip, and distribution
GRADLE_OPTS=-Xmx2048m ./gradlew \
           shadowJar \
           archiveZip \
           createDocs \
           createPackage \
           -Dspark.version=${SPARK_VERSION} \
           -Dspark.home=/spark \
           -Dtutorial.home=/usr/local/hail-tutorial-files \
           --gradle-user-home /gradle-cache

# update jar, zip, and distribution
GS_JAR=gs://hail-common/builds/${BRANCH_TARGET_NAME}/jars/hail-${BRANCH_TARGET_NAME}-${SHA}-Spark-${SPARK_VERSION}.jar
gsutil cp build/libs/hail-all-spark.jar ${GS_JAR}
gsutil acl set public-read ${GS_JAR}

GS_HAIL_ZIP=gs://hail-common/builds/${BRANCH_TARGET_NAME}/python/hail-${BRANCH_TARGET_NAME}-${SHA}.zip
gsutil cp build/distributions/hail-python.zip ${GS_HAIL_ZIP}
gsutil acl set public-read ${GS_HAIL_ZIP}

DISTRIBUTION=gs://hail-common/distributions/${BRANCH_TARGET_NAME}/Hail-${BRANCH_TARGET_NAME}-${SHA}-Spark-${SPARK_VERSION}.zip
gsutil cp build/distributions/hail.zip $DISTRIBUTION
gsutil acl set public-read $DISTRIBUTION

DOCS=gs://hail-common/builds/${BRANCH_TARGET_NAME}/docs/hail-${BRANCH_TARGET_NAME}-docs-${SHA}.tar.gz
tar cvf hail-${BRANCH_TARGET_NAME}-docs-${SHA}.tar.gz -C build/www docs
gsutil cp hail-${BRANCH_TARGET_NAME}-docs-${SHA}.tar.gz ${DOCS}
gsutil acl set public-read ${DOCS}

echo ${SHA} > latest-hash-spark-${SPARK_VERSION}.txt
gsutil cp ./latest-hash-spark-${SPARK_VERSION}.txt ${HASH_TARGET}
gsutil acl set public-read ${HASH_TARGET}
