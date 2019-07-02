#!/bin/bash
set -ex

LATEST_SHA=$(gsutil cat \
  gs://hail-common/builds/0.2/latest-hash/cloudtools-5-spark-2.4.0.txt 2>/dev/null || true)
if [[ $LATEST_SHA = "" ]]; then
    exit 0
fi

DEPLOYED_SHA=$(cat /var/www/0.2-deployed-hash.txt || true)
if [[ $DEPLOYED_SHA = $LATEST_SHA ]]; then
    exit 0
fi

mkdir -p /var/www/html-new

gsutil cat gs://hail-common/builds/0.2/docs/hail-0.2-docs-$LATEST_SHA.tar.gz |
    tar zxvf - -C /var/www/html-new --strip-components=1

ln -s /var/www/0.1 /var/www/html-new/docs/0.1

# just in case
rm -rf /var/www/html-old

mv /var/www/html /var/www/html-old || true
mv /var/www/html-new /var/www/html
rm -rf /var/www/html-old

echo $LATEST_SHA > /var/www/0.2-deployed-hash.txt
