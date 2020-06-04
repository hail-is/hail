#!/bin/bash
set -ex

LATEST_SHA=$(gsutil cat \
  gs://hail-common/builds/0.1/latest-hash-spark-2.0.2.txt 2>/dev/null || true)
if [[ $LATEST_SHA = "" ]]; then
    exit 0
fi

DEPLOYED_SHA=$(cat /var/www/0.1-deployed-hash.txt || true)
if [[ $DEPLOYED_SHA = $LATEST_SHA ]]; then
    exit 0
fi

web_root=/var/www/html
new=$(mktemp -d)
mkdir -p $new

gsutil cat gs://hail-common/builds/0.1/docs/hail-0.1-docs-$LATEST_SHA.tar.gz |
    tar xvf - -C $new --strip-components=2

old=$(mktemp -d)
mkdir -p $old

chown -R www-data $new
chmod -R u=rX,g=rX $new

mv $web_root/docs/0.1 $old || mkdir -p $web_root/docs
mv $new $web_root/docs/0.1
rm -rf $old

echo $LATEST_SHA > /var/www/0.1-deployed-hash.txt
