#!/bin/bash
set -ex

LATEST_SHA=$(gsutil cat \
  gs://hail-common/builds/0.1/latest-hash-spark-2.0.2.txt 2>/dev/null || true)
if [ "$LATEST_SHA" = "" ]; then
    exit 0
fi

DEPLOYED_SHA=$(cat /var/www/0.1-deployed-hash.txt || true)
if [ "$DEPLOYED_SHA" = "$LATEST_SHA" ]; then
    exit 0
fi

mkdir -p /var/www/0.1-new

gsutil cat gs://hail-common/builds/0.1/docs/hail-0.1-docs-$LATEST_SHA.tar.gz |
    tar xvf - -C /var/www/0.1-new --strip-components=2

# just in case
rm -rf /var/www/0.1-old

mv /var/www/0.1 /var/www/0.1-old || true
mv /var/www/0.1-new /var/www/0.1
rm -rf /var/www/0.1-old

echo $LATEST_SHA > /var/www/0.1-deployed-hash.txt
