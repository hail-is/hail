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

new=$(mktemp -d)

mkdir -p $new

gsutil cat gs://hail-common/builds/0.2/docs/hail-0.2-docs-$LATEST_SHA.tar.gz |
    tar zxvf - -C $new --strip-components=1

old=$(mktemp -d)

mv /var/www/docs/0.2 $old || true
mv $new /var/www/docs/0.2
rm -rf $old

echo $LATEST_SHA > /var/www/0.2-deployed-hash.txt
