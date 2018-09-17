set -ex

SPARK_VERSION=2.0.2
BRANCH_SOURCE_NAME=stable
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

echo ${SHA} > latest-hash-spark-${SPARK_VERSION}.txt
gsutil cp ./latest-hash-spark-${SPARK_VERSION}.txt ${HASH_TARGET}
gsutil acl set public-read ${HASH_TARGET}

# update website
## since we're non-interactive, we explicitly state the fingerprint for ci.hail.is
mkdir -p ~/.ssh
printf 'ci.hail.is ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQC3tuH5V3ubO7PqQ3gD2G7yFJ4bkwgSFNBqmaLmiuiCF86UpE4Lo4yQryt9VYssoLqsdStIOR0P/Bo3S4Nuj8cHCzAbft3/u25oa8lQKAazoiA0I82d7JXYurV/NvjH7O1MMuPohwjlBp+d4damUA3TO2oIHbYqzmArrvTs/k6DxUonWRRxZa0zW+edv78y6IdLXuSVyN5FPa+jWBMJar9CsvbsWUWtcJ8vHHldg0DJ7TFVecouy4U3hmQxi90OCGSk4N9vi+XC+EjoNeCmGt5/VGAnKCUZntOZluBqKKZ0/TWlC6HJgBWYQllnjAE1tFs9Xrrx+5ADB9quMtYVqk0R\n' \
       >> ~/.ssh/known_hosts

USER=web-updater
IDENTITY_FILE=/secrets/ci.hail.is-web-updater-rsa-key

rsync -rlv \
      -e "ssh -i ${IDENTITY_FILE}" \
      --exclude docs \
      --exclude misc \
      --exclude tools \
      build/www/ \
      ${USER}@ci.hail.is:/var/www/html/ \
      --delete

DEST=/var/www/html/docs/archive/${BRANCH_TARGET_NAME}/$SHA

ssh -i ${IDENTITY_FILE} \
    ${USER}@ci.hail.is \
    mkdir -p $DEST

scp -i ${IDENTITY_FILE} \
    -r build/www/docs/${BRANCH_SOURCE_NAME}/* \
    ${USER}@ci.hail.is:$DEST

ssh -i ${IDENTITY_FILE} \
    ${USER}@ci.hail.is \
    "rm -rf /var/www/html/docs/${BRANCH_TARGET_NAME} && \
     ln -s $DEST /var/www/html/docs/${BRANCH_TARGET_NAME} && \
     chgrp www-data $DEST /var/www/html/docs/${BRANCH_TARGET_NAME}"
