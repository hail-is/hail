#!/bin/bash
set -ex

cd hail

CLUSTER_NAME=ci-test-$(LC_CTYPE=C LC_ALL=C tr -dc 'a-z0-9' < /dev/urandom | head -c 8)

time source activate hail
time pip search cloudtools
time pip install -U cloudtools
gcloud auth activate-service-account \
    hail-ci-0-1@broad-ctsa.iam.gserviceaccount.com \
    --key-file=/secrets/hail-ci-0-1.key

on_exit() {
    trap "" INT TERM
    set +e
    rm -rf artifacts
    mkdir -p artifacts
    cp build/libs/hail-all-spark.jar artifacts/hail-all-spark.jar
    cp build/distributions/hail-python.zip artifacts/hail-python.zip
    cp -R build/www artifacts/www
    cp -R build/reports/tests artifacts/test-report
    cat <<EOF > artifacts/index.html
<html>
<body>
<h1>$(git rev-parse HEAD)</h1>
<ul>
<li><a href='hail-all-spark.jar'>hail-all-spark.jar</a></li>
<li><a href='hail-python.zip'>hail-python.zip</a></li>
<li><a href='www/index.html'>www/index.html</a></li>
<li><a href='test-report/index.html'>test-report/index.html</a></li>
</ul>
</body>
</html>
EOF
    time gcloud dataproc clusters delete ${CLUSTER_NAME} --async
}
trap on_exit EXIT

# some non-bash shells (in particular: dash and sh) do not trigger EXIT if the
# interpretation halts due to INT or TERM. Explicitly calling exit when INT or
# TERM is received ensures the EXIT handler is called.
trap "exit 42" INT TERM

GRADLE_OPTS=-Xmx2048m ./gradlew testAll makeDocs archiveZip --gradle-user-home /gradle-cache

time gsutil cp \
     build/libs/hail-all-spark.jar \
     gs://hail-ci-0-1/temp/$SOURCE_SHA/$TARGET_SHA/hail.jar

time gsutil cp \
     build/distributions/hail-python.zip \
     gs://hail-ci-0-1/temp/$SOURCE_SHA/$TARGET_SHA/hail.zip

time cluster start ${CLUSTER_NAME} \
     --version devel \
     --spark 2.2.0 \
     --bucket=hail-ci-0-1-dataproc-staging-bucket \
     --jar gs://hail-ci-0-1/temp/$SOURCE_SHA/$TARGET_SHA/hail.jar \
     --zip gs://hail-ci-0-1/temp/$SOURCE_SHA/$TARGET_SHA/hail.zip

time cluster submit ${CLUSTER_NAME} \
     cluster-sanity-check.py

time cluster stop ${CLUSTER_NAME}
