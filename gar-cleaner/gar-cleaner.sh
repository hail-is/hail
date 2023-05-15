#!/bin/bash

set -ex

gcloud artifacts docker images list us-docker.pkg.dev/hail-vdc/hail --include-tags > images2
cat images2 | sed 's/, /,/g' > images3
cat images3 | sed 's/  */ /g' > images4
cat images4 | awk -F' ' 'BEGIN { OFS = "\t" } { print $1, $2, $3, $4, $5 }' > images5
python3 find-expired-images.py $1
cat expired-images.csv
cat expired-images.csv | xargs -n1 -P16 gcloud artifacts docker images delete --delete-tags
gcloud artifacts repositories list
