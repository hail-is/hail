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

web_root=/var/www/html
new=$(mktemp -d)
mkdir -p $new

gsutil cat gs://hail-common/builds/0.2/docs/hail-0.2-docs-$LATEST_SHA.tar.gz |
    tar zxvf - -C $new --strip-components=1

old=$(mktemp -d)
mkdir -p $old

if [ -e "$new/docs" ]
then
    new="$new/docs"
fi

[ -e "$new/0.2" ]

chown -R www-data $new
chmod -R u=rX,g=rX $new

mv $web_root/docs/0.2 $old || mkdir -p $web_root/docs
mv $new/0.2 $web_root/docs/0.2

[ -e "$new/batch" ]

mv $web_root/docs/batch $old || true
mv $new/batch $web_root/docs/batch

rm -rf $old

echo $LATEST_SHA > /var/www/0.2-deployed-hash.txt
