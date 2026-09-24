#!/bin/bash

if [[ $# -ne 0 ]]
then
    cat <<EOF
Usage:
    test-gcp.sh
EOF
    exit -1
fi

set -ex

ID=$(cat /dev/urandom | LC_ALL=C tr -dc 'a-z0-9' | head -c 12)
CLUSTER=cluster-ci-$ID
MASTER=$CLUSTER-m
ZONE=us-central1-b

if ! (type gcloud > /dev/null); then
    # pick up gcloud
    PATH=$PATH:/usr/local/google-cloud-sdk/bin
fi

function cleanup {
    hailctl dataproc stop $CLUSTER
}
trap cleanup EXIT SIGINT

hailctl dataproc start $CLUSTER \
    --project broad-ctsa \
    --zone $ZONE \
    --subnet=default \
    --bucket=gs://hail-dataproc-staging-bucket-us-central1 \
    --temp-bucket=gs://hail-dataproc-temp-bucket-us-central1

# copy up necessary files
gcloud --project broad-ctsa compute scp \
       --zone $ZONE \
       ./out/hail/test/assembly.dest/out.jar \
       $MASTER:~/hail-test.jar

gcloud --project broad-ctsa compute ssh --zone ${ZONE} $MASTER -- 'mkdir -p hail/test'
gcloud --project broad-ctsa compute scp --recurse \
       --zone ${ZONE} \
       ./hail/test/resources \
       $MASTER:~/hail/test

set +e
cat <<'EOF' | gcloud --project broad-ctsa compute ssh --zone ${ZONE} $MASTER -- bash
set -ex

hdfs dfs -mkdir -p hail/test
hdfs dfs -rm -r -f -skipTrash hail/test/resources
hdfs dfs -put ./hail/test/resources hail/test

spark-submit \
  --class org.junit.platform.console.ConsoleLauncher \
  --jars ./hail-test.jar \
  --conf "spark.driver.extraClassPath=./hail-test.jar" \
  --conf 'spark.executor.extraClassPath=./hail-test.jar' \
  ./hail-test.jar \
  execute \
  --select-package=is.hail \
  --exclude-package=is.hail.shadedazure \
  --exclude-package=is.hail.io.fs \
  --exclude-package=is.hail.services \
  --fail-if-no-tests \
  --reports-dir=test-output
EOF
TEST_EXIT_CODE=$?
set -e

gcloud --project broad-ctsa compute scp --zone ${ZONE} --recurse $MASTER:test-output test-output

exit ${TEST_EXIT_CODE}
