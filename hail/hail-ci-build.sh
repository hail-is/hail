#!/bin/bash
set -ex

ROOT=$(cd .. && pwd)

CLUSTER_NAME=ci-test-$(LC_CTYPE=C LC_ALL=C tr -dc 'a-z0-9' < /dev/urandom | head -c 8)

time source activate hail
time pip search cloudtools
time pip install -U cloudtools
gcloud auth activate-service-account \
    hail-ci-0-1@broad-ctsa.iam.gserviceaccount.com \
    --key-file=/secrets/hail-ci-0-1.key
    
mkdir -p build

COMPILE_LOG="build/compilation.log"
SCALA_TEST_LOG="build/scala-test.log"
PYTHON_TEST_LOG="build/python-test.log"
DOCTEST_LOG="build/doctest.log"
GCP_LOG="build/gcp.log"

on_exit() {
    trap "" INT TERM
    set +e
    ARTIFACTS=${ROOT}/artifacts
    rm -rf ${ARTIFACTS}
    mkdir -p ${ARTIFACTS}
    cp build/libs/hail-all-spark.jar ${ARTIFACTS}/hail-all-spark.jar
    cp build/distributions/hail-python.zip ${ARTIFACTS}/hail-python.zip
    cp ${COMPILE_LOG} ${ARTIFACTS}
    cp ${SCALA_TEST_LOG} ${ARTIFACTS}
    cp ${PYTHON_TEST_LOG} ${ARTIFACTS}
    cp ${DOCTEST_LOG} ${ARTIFACTS}
    cp ${GCP_LOG} ${ARTIFACTS}
    cp -R build/www ${ARTIFACTS}/www
    cp -R build/reports/tests ${ARTIFACTS}/test-report
    cat <<EOF > ${ARTIFACTS}/index.html
<html>
<body>
<h1>$(git rev-parse HEAD)</h1>
<ul>
<li><a href='hail-all-spark.jar'>hail-all-spark.jar</a></li>
<li><a href='hail-python.zip'>hail-python.zip</a></li>
<li><a href='www/index.html'>www/index.html</a></li>
<li><a href='compilation.log'>Compilation log</a></li>
<li><a href='scala-test.log'>Scala test log></a/li>
<li><a href='python-test.log'>Python test log</a></li>
<li><a href='doctest.log'>Doctest log></a/li>
<li><a href='gcp.log'>GCP log</a></li>
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

export GRADLE_OPTS="-Xmx2048m"
export GRADLE_USER_HOME="/gradle-cache"

echo "Compiling..."
./gradlew shadowJar archiveZip > ${COMPILE_LOG}

test_project() {
    ./gradlew test > ${SCALA_TEST_LOG}
    ./gradlew testPython > ${PYTHON_TEST_LOG}
    ./gradlew doctest > ${DOCTEST_LOG}
}

test_gcp() {
    time gsutil cp \
         build/libs/hail-all-spark.jar \
         gs://hail-ci-0-1/temp/$SOURCE_SHA/$TARGET_SHA/hail.jar

    time gsutil cp \
         build/distributions/hail-python.zip \
         gs://hail-ci-0-1/temp/$SOURCE_SHA/$TARGET_SHA/hail.zip

    time cluster start ${CLUSTER_NAME} \
         --version devel \
         --spark 2.2.0 \
         --max-idle 40m \
         --bucket=hail-ci-0-1-dataproc-staging-bucket \
         --jar gs://hail-ci-0-1/temp/$SOURCE_SHA/$TARGET_SHA/hail.jar \
         --zip gs://hail-ci-0-1/temp/$SOURCE_SHA/$TARGET_SHA/hail.zip \
         --vep

    time cluster submit ${CLUSTER_NAME} \
         cluster-sanity-check.py

    time cluster submit ${CLUSTER_NAME} \
         cluster-vep-check.py

    time cluster stop ${CLUSTER_NAME} --async
}

test_project &
TEST_PROJECT_PID=$!

test_gcp > ${GCP_LOG} &
TEST_GCP_PID=$!

for pid in "$TEST_PROJECT_PID $TEST_GCP_PID"; do
    wait $pid;
done
