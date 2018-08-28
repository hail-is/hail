set -ex

SPARK_VERSION=2.2.0
BRANCH=devel

source activate hail

# build jar, zip, and distribution
GRADLE_OPTS=-Xmx2048m ./gradlew \
           shadowJar \
           archiveZip \
           makeDocsNoTest \
           createPackageNoTest \
           --gradle-user-home /gradle-cache
SHA=$(git rev-parse --short=12 HEAD)

# update jar, zip, and distribution
GS_JAR=gs://hail-common/${BRANCH}/jars/hail-${BRANCH}-${SHA}-Spark-${SPARK_VERSION}.jar
gsutil cp build/libs/hail-all-spark.jar ${GS_JAR}
gsutil acl set public-read ${GS_JAR}

GS_HAIL_ZIP=gs://hail-common/builds/${BRANCH}/python/hail-${BRANCH}-${SHA}.zip
gsutil cp build/libs/hail-all-spark.jar ${GS_HAIL_ZIP}
gsutil acl set public-read ${GS_HAIL_ZIP}

DISTRIBUTION=gs://hail-common/distributions/${BRANCH}/Hail-${BRANCH}-${SHA}-Spark-${SPARK_VERSION}.zip
gsutil cp build/distributions/hail.zip $DISTRIBUTION
gsutil acl set public-read $DISTRIBUTION

echo ${SHA} > latest-hash-spark${SPARK_VERSION}.txt
HASH_TARGET=gs://hail-common/builds/${BRANCH}/latest-hash-spark-${SPARK_VERSION}.txt
gsutil cp ./latest-hash-spark${SPARK_VERSION}.txt ${HASH_TARGET}
gsutil acl set public-read ${HASH_TARGET}

# update website
## since we're interactive, we explicitly state the fingerprint for ci.hail.is
mkdir -p ~/.ssh
printf 'ci.hail.is ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQC3tuH5V3ubO7PqQ3gD2G7yFJ4bkwgSFNBqmaLmiuiCF86UpE4Lo4yQryt9VYssoLqsdStIOR0P/Bo3S4Nuj8cHCzAbft3/u25oa8lQKAazoiA0I82d7JXYurV/NvjH7O1MMuPohwjlBp+d4damUA3TO2oIHbYqzmArrvTs/k6DxUonWRRxZa0zW+edv78y6IdLXuSVyN5FPa+jWBMJar9CsvbsWUWtcJ8vHHldg0DJ7TFVecouy4U3hmQxi90OCGSk4N9vi+XC+EjoNeCmGt5/VGAnKCUZntOZluBqKKZ0/TWlC6HJgBWYQllnjAE1tFs9Xrrx+5ADB9quMtYVqk0R\n' \
  >> ~/.ssh/known_hosts
USER=web-updater
rsync -rlv \
      -e "ssh -i /secrets/ci.hail.is-web-updater-rsa-key" \
      --exclude docs \
      --exclude misc \
      --exclude tools \
      build/www/ \
      ${USER}@ci.hail.is:/var/www/html/ \
      --delete
DEST=/var/www/html/docs/archive/${BRANCH}/$SHA
ssh -i /secrets/ci.hail.is-web-updater-rsa-key \
    ${USER}@ci.hail.is \
    mkdir -p $DEST
scp -i /secrets/ci.hail.is-web-updater-rsa-key \
    -r build/www/docs/${BRANCH}/* \
    ${USER}@ci.hail.is:$DEST
ssh -i /secrets/ci.hail.is-web-updater-rsa-key \
    ${USER}@ci.hail.is \
    "rm -rf /var/www/html/docs/${BRANCH} && \
     ln -s $DEST /var/www/html/docs/${BRANCH} && \
     chown :www-data $DEST /var/www/html/docs/${BRANCH}"
