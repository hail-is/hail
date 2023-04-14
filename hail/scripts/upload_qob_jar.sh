#!/bin/bash

set -ex

NAMESPACE=$1
REVISION=$2
SHADOW_JAR=$3
PATH_FILE=$4


TOKEN=$(cat /dev/urandom 2> /dev/null | LC_ALL=C tr -dc 'a-z0-9' 2> /dev/null | head -c 12)
QUERY_STORAGE_URI=https://haildevbatch.blob.core.windows.net/query
TEST_STORAGE_URI=$(kubectl get secret global-config --template={{.data.test_storage_uri}} | base64 --decode)

if [[ "${NAMESPACE}" == "default" ]]; then
    if [[ "${UPLOAD_RELEASE_JAR}" == "true" ]]; then
	JAR_LOCATION="${QUERY_STORAGE_URI}/jars/${REVISION}.jar"
    else
	JAR_LOCATION="${QUERY_STORAGE_URI}/jars/$(whoami)/${TOKEN}/${REVISION}.jar"
    fi
else
    JAR_LOCATION="${TEST_STORAGE_URI}/${NAMESPACE}/jars/${TOKEN}/${REVISION}.jar"
fi

# gcloud storage cp ${SHADOW_JAR} ${JAR_LOCATION}
azcopy copy ${SHADOW_JAR} ${JAR_LOCATION}
echo ${JAR_LOCATION} > ${PATH_FILE}
