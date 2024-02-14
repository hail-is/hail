#!/bin/bash

set -ex

cluster_name=cluster-$(whoami)-$(LC_ALL=C tr -dc '[:lower:]' </dev/urandom | head -c 6)

stop_dataproc () {
    exit_code=$?

    set +e

    hailctl dataproc stop $cluster_name || true  # max-idle or max-age will delete it

    exit $exit_code
}
trap stop_dataproc EXIT

if [ $1 == "GRCh37" ]
then
    EXCLUDE="GRCh38"
fi

if [ $1 == "GRCh38" ]
then
    EXCLUDE="GRCh37"
fi

cluster_test_files=$(ls python/cluster-tests/*.py | grep -ve "python/cluster-tests/cluster-vep-check-$EXCLUDE.py")

hailctl dataproc \
        start $cluster_name \
        --max-idle 10m \
        --max-age 120m \
        --vep $1 \
        --num-preemptible-workers=4 \
        --requester-pays-allow-buckets hail-us-central1-vep,hail-1kg \
        --subnet=default \
	--bucket=hail-dataproc-staging-bucket-us-central1 \
	--temp-bucket=hail-dataproc-temp-bucket-us-central1
for file in $cluster_test_files
do
    hailctl dataproc \
            submit \
            $cluster_name $file
done
