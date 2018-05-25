#!/bin/sh

set -e

mkdir -p build/tmp/python/hail/docs

TARGET=build/tmp/python/hail/docs/distLinks.rst
rm -f $TARGET

HAIL_VERSION=$1
HASH=$(git rev-parse --short=12 HEAD)


echo "Hail uploads distributions to Google Storage as part of our continuous integration suite." >> $TARGET

echo "You can download a pre-built distribution from the below links. Make sure you download the distribution that matches your Spark version!" >> $TARGET

echo "" >> $TARGET

shift
while test ${#} -gt 0
do
  echo "- \`Current distribution for Spark $1 <https://storage.googleapis.com/hail-common/distributions/$HAIL_VERSION/Hail-$HAIL_VERSION-$HASH-Spark-$1.zip>\`_" >> $TARGET
  shift
done
