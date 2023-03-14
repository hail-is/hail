#!/bin/bash

if [[ $# -ne 1 ]]
then
    cat <<EOF
Usage:
    test-gcp.sh DATAPROC_VERSION
EOF
    exit -1
fi

set -ex

DATAPROC_VERSION=$1
ID=$(cat /dev/urandom | LC_ALL=C tr -dc 'a-z0-9' | head -c 12)
CLUSTER=cluster-ci-$ID
MASTER=$CLUSTER-m
ZONE=us-central1-b

if ! (type gcloud > /dev/null); then
    # pick up gcloud
    PATH=$PATH:/usr/local/google-cloud-sdk/bin
fi

function cleanup {
  gcloud --project broad-ctsa -q dataproc clusters delete --async $CLUSTER
}
trap cleanup EXIT SIGINT

gcloud --project broad-ctsa dataproc clusters create $CLUSTER \
    --zone $ZONE \
    --master-machine-type n1-standard-2 \
    --master-boot-disk-size 100 \
    --num-workers 2 \
    --worker-machine-type n1-standard-2 \
    --worker-boot-disk-size 100 \
    --image-version ${DATAPROC_VERSION} \
    --initialization-actions 'gs://hail-dataproc-deps/initialization-actions.sh'

# copy up necessary files
gcloud --project broad-ctsa compute scp \
       --zone $ZONE \
       ./build/libs/hail-all-spark-test.jar \
       ./testng.xml \
       $MASTER:~

gcloud --project broad-ctsa compute ssh --zone ${ZONE} $MASTER -- 'mkdir -p src/test'
gcloud --project broad-ctsa compute scp --recurse \
       --zone ${ZONE} \
       ./src/test/resources \
       $MASTER:~/src/test

set +e
cat <<'EOF' | gcloud --project broad-ctsa compute ssh --zone ${ZONE} $MASTER -- bash
set -ex

hdfs dfs -mkdir -p src/test
hdfs dfs -rm -r -f -skipTrash src/test/resources
hdfs dfs -put ./src/test/resources src/test

spark-submit \
  --class org.testng.TestNG \
  --jars ./hail-all-spark-test.jar \
  --conf "spark.driver.extraClassPath=./hail-all-spark-test.jar" \
  --conf 'spark.executor.extraClassPath=./hail-all-spark-test.jar' \
  ./hail-all-spark-test.jar ./testng.xml
EOF
TEST_EXIT_CODE=$?
set -e

gcloud --project broad-ctsa compute scp --zone ${ZONE} --recurse $MASTER:test-output test-output

exit ${TEST_EXIT_CODE}
