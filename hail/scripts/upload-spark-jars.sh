#!/usr/bin/env bash

set -exo pipefail

NAMESPACE=$1
SPARK_VERSION=$2

# JAR_PREFIX may be defined in the environment (as in ci builds, where
# kubectl is unavailable); otherwise derive it from the namespace's
# global-config secret.
if [[ -z $JAR_PREFIX ]]; then
    if [[ $NAMESPACE == "default" ]]; then
        JAR_PREFIX=$(kubectl get secret global-config --template={{.data.query_storage_uri}} | base64 --decode)
    else
        BUCKET=$(kubectl get secret global-config --template={{.data.test_storage_uri}} | base64 --decode)
        JAR_PREFIX="${BUCKET}/${NAMESPACE}"
    fi
fi

ARCHIVE_LOCATION="${JAR_PREFIX}/spark/spark-${SPARK_VERSION}.tar.gz"

EXISTS=$(python3 -c "import hailtop.fs as fs; print(fs.exists('${ARCHIVE_LOCATION}'))")

if [[ $EXISTS == "False" ]]; then
    TMPDIR=$(mktemp -d)
    trap 'rm -rf ${TMPDIR}' EXIT

    # the pyspark sdist ships the jars that the spark distribution of the same
    # version runs against, which is what jars compiled against that spark expect
    pip download "pyspark==${SPARK_VERSION}" --no-deps --no-binary :all: -d "${TMPDIR}"
    tar -xzf "${TMPDIR}/pyspark-${SPARK_VERSION}.tar.gz" \
        -C "${TMPDIR}" \
        "pyspark-${SPARK_VERSION}/deps/jars"
    tar -czf "${TMPDIR}/spark-${SPARK_VERSION}.tar.gz" \
        -C "${TMPDIR}/pyspark-${SPARK_VERSION}/deps/jars" \
        .

    python3 -m hailtop.aiotools.copy \
        -vvv \
        'null' \
        '[{"from":"'${TMPDIR}/spark-${SPARK_VERSION}.tar.gz'", "to":"'${ARCHIVE_LOCATION}'"}]' \
        --timeout 600
fi

echo "${ARCHIVE_LOCATION}"
