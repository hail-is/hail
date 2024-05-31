#!/bin/bash

set -ex

NAMESPACE=$1
REVISION=$2
SHADOW_JAR=$3
PATH_FILE=$4


if [[ "${NAMESPACE}" == "default" ]]; then
    JAR_PREFIX=$(kubectl get secret global-config --template={{.data.query_storage_uri}} | base64 --decode)
else
    BUCKET=$(kubectl get secret global-config --template={{.data.test_storage_uri}} | base64 --decode)
    JAR_PREFIX="${BUCKET}/${NAMESPACE}"
fi

if [[ "${UPLOAD_RELEASE_JAR}" == "true" ]]; then
    JAR_DIR="jars"
else
    JAR_DIR="jars/dev"
fi

JAR_LOCATION="${JAR_PREFIX}/${JAR_DIR}/${REVISION}.jar"

python3 -m hailtop.aiotools.copy \
	-vvv \
	'null' \
	'[{"from":"'${SHADOW_JAR}'", "to":"'${JAR_LOCATION}'"}]' \
	--timeout 600
echo ${JAR_LOCATION} > ${PATH_FILE}
