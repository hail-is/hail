#!/bin/sh

mkdir -p python/hail/docs
rm -f python/hail/docs/distLinks.rst

HAIL_VERSION=$1
HASH=$(git rev-parse --short HEAD)


echo "Hail uploads distributions to Google Storage as part of our continuous integration suite." >> python/hail/docs/distLinks.rst

echo "You can download a pre-built distribution from the below links. Make sure you download the distribution that matches your Spark version!" >> python/hail/docs/distLinks.rst

echo "" >> python/hail/docs/distLinks.rst

shift
while test ${#} -gt 0
do
  echo "- \`Current distribution for Spark $1 <https://storage.googleapis.com/hail-common/distributions/Hail-$HAIL_VERSION-$HASH-Spark-$1.zip>\`_" >> python/hail/docs/distLinks.rst
  shift
done
