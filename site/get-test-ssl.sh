#!/bin/bash

mkdir -p ssl-config-site

f=$(mktemp)

kubectl get secret ssl-config-site -o json > $f

for x in $(jq -r '.data | keys[]' $f)
do
    jq -r '.data["'$x'"]' $f | base64 -D >ssl-config-site/$x
done
