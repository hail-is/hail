#!/bin/bash

set -ex

NAMESPACE=$1
REVISION=$2
SHADOW_JAR=$3
PATH_FILE=$4


TOKEN=$(cat /dev/urandom 2> /dev/null | LC_ALL=C tr -dc 'a-z0-9' 2> /dev/null | head -c 12)
QUERY_STORAGE_URI=$(kubectl get secret global-config --template={{.data.query_storage_uri}} | base64 --decode)
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

az storage blob upload --file ${SHADOW_JAR} --account-name haildevbatch --container-name query --name jars/dking/lfrxrx935eaq/74f44922afbdceb900661e28011ed4004e6ecb25.jar
echo ${JAR_LOCATION} > ${PATH_FILE}
