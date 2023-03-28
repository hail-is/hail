#!/bin/bash

set -ex

NAMESPACE=$1
REVISION=$2
SHADOW_JAR=$3
PATH_FILE=$4



QUERY_STORAGE_URI=$(kubectl get secret global-config --template={{.data.query_storage_uri}} | base64 --decode)
TEST_STORAGE_URI=$(kubectl get secret global-config --template={{.data.test_storage_uri}} | base64 --decode)

if [[ "${NAMESPACE}" == "default" ]]; then
  if [[ "${UPLOAD_RELEASE_JAR}" == "true" ]]; then
    JAR_LOCATION="${QUERY_STORAGE_URI}/jars/${REVISION}.jar"
    else
    JAR_LOCATION="${TEST_STORAGE_URI}/${NAMESPACE}/jars/${TOKEN}/${REVISION}.jar"
  fi
  else
    JAR_LOCATION="${TEST_STORAGE_URI}/${NAMESPACE}/jars/${TOKEN}/${REVISION}.jar"
fi

gcloud storage cp ${SHADOW_JAR} ${JAR_LOCATION}
echo ${JAR_LOCATION} > ${PATH_FILE}
