#!/bin/bash

set -ex

cluster_name_37=cluster-$(whoami)-$(LC_ALL=C tr -dc '[:lower:]' </dev/urandom | head -c 6)
cluster_name_38=cluster-$(whoami)-$(LC_ALL=C tr -dc '[:lower:]' </dev/urandom | head -c 6)

stop_dataproc () {
    exit_code=$?

    set +e

    hailctl dataproc stop $cluster_name_37
    hailctl dataproc stop $cluster_name_38

    exit $exit_code
}
trap stop_dataproc EXIT

cluster_37_test_files=$(ls python/cluster-tests/*.py | grep -ve 'python/cluster-tests/cluster-vep-check-GRCh38.py')
cluster_38_test_files=$(ls python/cluster-tests/*.py | grep -ve 'python/cluster-tests/cluster-vep-check-GRCh37.py')

hailctl dataproc start $cluster_name_37 --max-idle 10m --vep GRCh37 --requester-pays-allow-buckets hail-us-vep
for file in $cluster_37_test_files
do
    hailctl dataproc submit $cluster_name_37 $file
done

hailctl dataproc start $cluster_name_38 --max-idle 10m --vep GRCh38 --requester-pays-allow-buckets hail-us-vep
for file in $cluster_38_test_files
do
    hailctl dataproc submit $cluster_name_38 $filen
done
