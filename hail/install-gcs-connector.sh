#!/bin/bash

which gcloud >/dev/null \
    || (echo 'install `gcloud` first https://cloud.google.com/sdk/docs/quickstarts' ; exit 1)

gcloud auth list | grep -Ee '^\* ' | grep -vq 'compute@developer.gserviceaccount.com' \
    || (echo "WARNING: you are logged in as $(gcloud auth list | grep -Ee '^\* '), if this fails, try logging in as YOU@YOUR_ORGANIZATION.com")

set -e

COMPUTE_ENGINE_SERVICE_ACCOUNT=$(gcloud iam service-accounts list \
                                     | grep -e 'Compute Engine default service account' \
                                     | sed 's:Compute Engine default service account[ ]*::' | awk '{print $1}')

mkdir -p ${HOME}/.hail/gcs-keys/

KEY_FILE=${HOME}/.hail/gcs-keys/gcs-connector-key.json
gcloud iam service-accounts keys create \
       ${KEY_FILE} \
       --iam-account $COMPUTE_ENGINE_SERVICE_ACCOUNT
chmod 440 ${KEY_FILE}

which find_spark_home.py >/dev/null \
    || (echo 'you do not have find_spark_home.py on your path, did you already `pip install hail`?' ; exit 1)

SPARK_HOME=$(find_spark_home.py)

set +e
mkdir ${SPARK_HOME}/conf
set -e

CONF_FILE=${SPARK_HOME}/conf/spark-defaults.conf

SVC_ACCT_ENABLE=spark.hadoop.google.cloud.auth.service.account.enable
SVC_ACCT_KEY_FILE=spark.hadoop.google.cloud.auth.service.account.json.keyfile

if [ -e ${CONF_FILE} ]; then
    sed -i '' "/${SVC_ACCT_ENABLE}/d" ${CONF_FILE}
    sed -i '' "/${SVC_ACCT_KEY_FILE}/d" ${CONF_FILE}
fi
echo "${SVC_ACCT_ENABLE} true" >> ${CONF_FILE}
echo "${SVC_ACCT_KEY_FILE} ${KEY_FILE}" >> ${CONF_FILE}

curl -v \
     https://storage.googleapis.com/hadoop-lib/gcs/gcs-connector-hadoop2-latest.jar \
     > ${SPARK_HOME}/jars/gcs-connector-hadoop2-latest.jar

echo "success"
