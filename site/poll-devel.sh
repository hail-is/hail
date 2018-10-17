#!/bin/bash
set -ex

LATEST_SHA=$(gsutil cat \
  gs://hail-common/builds/devel/latest-hash/cloudtools-2-spark-2.2.0.txt 2>/dev/null || true)
if [[ $LATEST_SHA = "" ]]; then
    exit 0
fi

DEPLOYED_SHA=$(cat /var/www/devel-deployed-hash.txt || true)
if [[ $DEPLOYED_SHA = $LATEST_SHA ]]; then
    exit 0
fi

mkdir -p /var/www/html-new

gsutil cat gs://hail-common/builds/devel/docs/hail-devel-docs-$LATEST_SHA.tar.gz |
    tar xvf - -C /var/www/html-new --strip-components=1

ln -s /var/www/0.1 /var/www/html-new/docs/0.1
ln -s /var/www/html/docs/devel /var/www/html-new/docs/0.2
ln -s /var/www/html/docs/devel /var/www/html-new/docs/stable

# just in case
rm -rf /var/www/html-old

mv /var/www/html /var/www/html-old || true
mv /var/www/html-new /var/www/html
rm -rf /var/www/html-old

echo $LATEST_SHA > /var/www/devel-deployed-hash.txt
