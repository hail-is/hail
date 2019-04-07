#!/bin/bash
set -ex

export CXXFLAGS="${CXXFLAGS} -Werror"

ROOT=$(cd .. && pwd)

CLUSTER_NAME=ci-test-$(LC_CTYPE=C LC_ALL=C tr -dc 'a-z0-9' < /dev/urandom | head -c 8)

time source activate hail
time pip search cloudtools
time pip install -U cloudtools
gcloud auth activate-service-account \
    hail-ci-0-1@broad-ctsa.iam.gserviceaccount.com \
    --key-file=/secrets/hail-ci-0-1.key
gcloud config set project broad-ctsa

mkdir -p build

COMPILE_LOG="build/compilation.log"
CXX_TEST_LOG="build/cxx-test.log"
SCALA_TEST_LOG="build/scala-test.log"
CXX_CODEGEN_TEST_LOG="build/codegen-test.log"
PYTHON_TEST_LOG="build/python-test.log"
DOCTEST_LOG="build/doctest.log"
DOCS_LOG="build/docs.log"
GCP_LOG="build/gcp.log"
PIP_PACKAGE_LOG="build/pip-package.log"

COMP_SUCCESS="build/_COMP_SUCCESS"
CXX_TEST_SUCCESS="build/_CXX_TEST_SUCCESS"
SCALA_TEST_SUCCESS="build/_SCALA_TEST_SUCCESS"
CXX_CODEGEN_TEST_SUCCESS="build/_CXX_CODEGEN_TEST_SUCCESS"
PYTHON_TEST_SUCCESS="build/_PYTHON_TEST_SUCCESS"
DOCTEST_SUCCESS="build/_DOCTEST_SUCCESS"
DOCS_SUCCESS="build/_DOCS_SUCCESS"
GCP_SUCCESS="build/_GCP_SUCCESS"
GCP_STOPPED="build/_GCP_STOPPED"
PIP_PACKAGE_SUCCESS="build/_PIP_PACKAGE_SUCCESS"
PIP_PACKAGE_STOPPED="build/_PIP_PACKAGE_STOPPED"

SUCCESS='<span style="color:green;font-weight:bold">SUCCESS</span>'
FAILURE='<span style="color:red;font-weight:bold">FAILURE</span>'
SKIPPED='<span style="color:gray;font-weight:bold">SKIPPED</span>'
STOPPED='<span style="color:gray;font-weight:bold">STOPPED</span>'

get_status() {
    FILE_LOC=$1
    DEPENDENCY=$2
    if [ -n "${DEPENDENCY}" ] && [ "${DEPENDENCY}" != "${SUCCESS}" ]; then
        echo ${SKIPPED};
    elif [ -e ${FILE_LOC} ]; then
        echo ${SUCCESS};
    else echo ${FAILURE};
    fi
}


on_exit() {
    trap "" INT TERM
    set +e
    ARTIFACTS=${ROOT}/artifacts
    rm -rf ${ARTIFACTS}
    mkdir -p ${ARTIFACTS}
    cp build/libs/hail-all-spark.jar ${ARTIFACTS}/hail-all-spark.jar
    cp build/distributions/hail-python.zip ${ARTIFACTS}/hail-python.zip
    cp ${COMPILE_LOG} ${ARTIFACTS}
    cp ${CXX_TEST_LOG} ${ARTIFACTS}
    cp ${SCALA_TEST_LOG} ${ARTIFACTS}
    cp ${CXX_CODEGEN_TEST_LOG} ${ARTIFACTS}
    cp ${PYTHON_TEST_LOG} ${ARTIFACTS}
    cp ${DOCS_LOG} ${ARTIFACTS}
    cp ${DOCTEST_LOG} ${ARTIFACTS}
    cp ${GCP_LOG} ${ARTIFACTS}
    cp ${PIP_PACKAGE_LOG} ${ARTIFACTS}
    cp -R build/www ${ARTIFACTS}/www
    cp -R src/main/c/build/reports ${ARTIFACTS}/cxx-report
    cp -R build/reports/tests ${ARTIFACTS}/test-report
    cp -R build/reports/codegen-tests ${ARTIFACTS}/codegen-test-report
    cp -R build/reports/pytest.html ${ARTIFACTS}/hail-python-test.html

    COMP_STATUS=$(get_status "${COMP_SUCCESS}")
    CXX_TEST_STATUS=$(get_status "${CXX_TEST_SUCCESS}")
    SCALA_TEST_STATUS=$(get_status "${SCALA_TEST_SUCCESS}" "${CXX_TEST_STATUS}")
    CXX_CODEGEN_TEST_STATUS=$(get_status "${CXX_CODEGEN_TEST_SUCCESS}" "${SCALA_TEST_STATUS}")
    PYTHON_TEST_STATUS=$(get_status "${PYTHON_TEST_SUCCESS}" "${CXX_CODEGEN_TEST_STATUS}")
    DOCTEST_STATUS=$(get_status "${DOCTEST_SUCCESS}" "${PYTHON_TEST_STATUS}")
    DOCS_STATUS=$(get_status "${DOCS_SUCCESS}" "${DOCTEST_STATUS}")
    GCP_STATUS=$(if [ -e ${GCP_STOPPED} ]; then echo "${STOPPED}"; else get_status "${GCP_SUCCESS}"; fi)
    PIP_PACKAGE_STATUS=$(if [ -e ${PIP_PACKAGE_STOPPED} ]; then echo "${STOPPED}"; else get_status "${PIP_PACKAGE_SUCCESS}"; fi)

    cat <<EOF > ${ARTIFACTS}/index.html
<html>
<head>
<style type="text/css">
    body {
        font-family: "Lucida Sans Unicode", "Lucida Grande", sans-serif;
    }
</style>
</head>
<h2>$(git rev-parse HEAD)</h2>
<h3>Compilation</h3>
<table>
<tbody>
<tr>
<td>${COMP_STATUS}</td>
<td><a href='compilation.log'>Compilation log</a></td>
</tr>
<tr>
<td>${COMP_STATUS}</td>
<td><a href='hail-all-spark.jar'>hail-all-spark.jar</a></td>
</tr>
<tr>
<td>${COMP_STATUS}</td>
<td><a href='hail-python.zip'>hail-python.zip</a></td>
</tr>
</tbody>
</table>
<h3>Tests</h3>
<table>
<tbody>
<tr>
<td>${CXX_TEST_STATUS}</td>
<td><a href='cxx-test.log'>C++ test log</a></td>
</tr>
<tr>
<td>${CXX_TEST_STATUS}</td>
<td><a href='cxx-report/index.html'>Catch2 report</a></td>
</tr>
<tr>
<td>${SCALA_TEST_STATUS}</td>
<td><a href='scala-test.log'>Scala test log</a></td>
</tr>
<tr>
<td>${SCALA_TEST_STATUS}</td>
<td><a href='test-report/index.html'>TestNG report</a></td>
</tr>
<td>${CXX_CODEGEN_TEST_STATUS}</td>
<td><a href='codegen-test.log'>Scala test log (C++ Codegen tests)</a></td>
</tr>
<tr>
<td>${CXX_CODEGEN_TEST_STATUS}</td>
<td><a href='codegen-test-report/index.html'>TestNG report (C++ Codegen tests)</a></td>
</tr>
<tr>
<td>${PYTHON_TEST_STATUS}</td>
<td><a href='python-test.log'>PyTest log</a></td>
</tr>
<tr>
<td>${PYTHON_TEST_STATUS}</td>
<td><a href='hail-python-test.html'>PyTest report</a></td>
</tr>
<tr>
<td>${DOCTEST_STATUS}</td>
<td><a href='doctest.log'>Doctest log</a/td>
</tr>
</tbody>
</table>
<h3>Docs</h3>
<table>
<tbody>
<tr>
<td>${DOCS_STATUS}</td>
<td><a href='docs.log'>Docs build log</a/td>
</tr>
<tr>
<td>${DOCS_STATUS}</td>
<td><a href='www/index.html'>Generated website</a></td>
</tr>
<tr>
<td>${DOCS_STATUS}</td>
<td><a href='www/docs/0.2/index.html'>Generated docs</a></td>
</tr>
</tbody>
</table>
<h3>Cloud test</h3>
<table>
<tbody>
<tr>
<td>${GCP_STATUS}</td>
<td><a href='gcp.log'>GCP log</a></td>
</tr>
</tbody>
</table>
<h3>PIP Package test</h3>
<table>
<tbody>
<tr>
<td>${PIP_PACKAGE_STATUS}</td>
<td><a href='pip-package.log'>PIP Package log</a></td>
</tr>
</tbody>
</table>
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
./gradlew shadowJar archiveZip > ${COMPILE_LOG} 2>&1
touch ${COMP_SUCCESS}

test_project() {
    ./gradlew nativeLibTest > ${CXX_TEST_LOG} 2>&1
    touch ${CXX_TEST_SUCCESS}
    ./gradlew test > ${SCALA_TEST_LOG} 2>&1
    touch ${SCALA_TEST_SUCCESS}
    ./gradlew testCppCodegen > ${CXX_CODEGEN_TEST_LOG} 2>&1
    touch ${CXX_CODEGEN_TEST_SUCCESS}
    ./gradlew testPython > ${PYTHON_TEST_LOG} 2>&1
    touch ${PYTHON_TEST_SUCCESS}
    ./gradlew doctest > ${DOCTEST_LOG} 2>&1
    touch ${DOCTEST_SUCCESS}
    ./gradlew makeDocs > ${DOCS_LOG} 2>&1
    touch ${DOCS_SUCCESS}
}

test_gcp() {
    local conf_file="./hail-config-0.2-test.json"
    python ./create_config_file.py '0.2' $conf_file
    time gsutil cp \
         build/libs/hail-all-spark.jar \
         gs://hail-ci-0-1/temp/$SOURCE_SHA/$TARGET_SHA/hail.jar

    time gsutil cp \
         build/distributions/hail-python.zip \
         gs://hail-ci-0-1/temp/$SOURCE_SHA/$TARGET_SHA/hail.zip

    time cluster start ${CLUSTER_NAME} \
         --master-machine-type n1-standard-1 \
         --master-boot-disk-size 40 \
         --worker-machine-type n1-standard-1 \
         --worker-boot-disk-size 40 \
         --version 0.2 \
         --spark 2.2.0 \
         --max-idle 10m \
         --bucket hail-ci-0-1-dataproc-staging-bucket \
         --config-file $conf_file \
         --hash $(git rev-parse --short=12 HEAD) \
         --jar gs://hail-ci-0-1/temp/$SOURCE_SHA/$TARGET_SHA/hail.jar \
         --zip gs://hail-ci-0-1/temp/$SOURCE_SHA/$TARGET_SHA/hail.zip \
         --vep

    for script in python/cluster-tests/**.py; do
        time cluster submit ${CLUSTER_NAME} $script
    done

    time cluster stop ${CLUSTER_NAME} --async
    touch ${GCP_SUCCESS}
}

test_pip_package() {
    ./gradlew shadowJar
    cp build/libs/hail-all-spark.jar python/hail/hail-all-spark.jar
    cp ../README.md python/
    CONDA_ENV_NAME=$(LC_CTYPE=C LC_ALL=C tr -dc 'a-z0-9' < /dev/urandom | head -c 8)
    conda create -n $CONDA_ENV_NAME python=3.7
    conda activate $CONDA_ENV_NAME
    pip install ./python
    time env -u SPARK_HOME python -c 'import hail as hl; hl.init(); hl.balding_nichols_model(3,100,100)._force_count_rows()'
    # FIXME: also test on Mac OS X
    touch ${PIP_PACKAGE_SUCCESS}
}

test_project &
PID1=$!

test_gcp > ${GCP_LOG} 2>&1 &
PID2=$!

test_pip_package > ${PIP_PACKAGE_LOG} 2>&1 &
PID3=$!

wait $PID1
if [ "$?" != "0" ]; then
    echo "test_project failed!"
    if [ -ne ${GCP_SUCCESS} ]; then
        touch ${GCP_STOPPED}
    fi
    if [ -ne ${PIP_PACKAGE_SUCCESS} ]; then
        touch ${PIP_PACKAGE_STOPPED}
    fi
    exit 1
fi

wait $PID2
if [ "$?" != "0" ]; then
    echo "test_gcp failed!"
    if [ -ne ${PIP_PACKAGE_SUCCESS} ]; then
        touch ${PIP_PACKAGE_STOPPED}
    fi
    exit 1
fi

wait $PID3
if [ "$?" != "0" ]; then
    echo "test_pip_package failed!"
    exit 1
fi
