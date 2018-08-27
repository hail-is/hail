set -ex
source activate hail
GRADLE_OPTS=-Xmx2048m ./gradlew shadowJar archiveZip makeDocsNoTest --gradle-user-home /gradle-cache
SHA=$(git rev-parse --short=12 HEAD)
gsutil cp build/libs/hail-all-spark.jar gs://hail-common/devel/jars/hail-devel-${SHA}-Spark-2.2.0.jar
gsutil cp build/libs/hail-all-spark.jar gs://hail-common/devel/python/hail-devel-${SHA}.zip
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
DEST=/var/www/html/docs/archive/devel/$SHA
ssh -i /secrets/ci.hail.is-web-updater-rsa-key \
    ${USER}@ci.hail.is \
    mkdir -p $DEST
scp -i /secrets/ci.hail.is-web-updater-rsa-key \
    -r build/www/docs/devel/* \
    ${USER}@ci.hail.is:$DEST
ssh -i /secrets/ci.hail.is-web-updater-rsa-key \
    ${USER}@ci.hail.is \
    "rm -rf /var/www/html/docs/devel && \
     ln -s $DEST /var/www/html/docs/devel && \
     chown :www-data $DEST /var/www/html/docs/devel"
