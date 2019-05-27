#!/bin/bash
set -ex

unset SPARK_MASTER_HOST
unset SPARK_MASTER_PORT

echo "0.0.0.0 spark-master" >> /etc/hosts

export SPARK_DAEMON_JAVA_OPTS="-Djava.net.preferIPv4Stack=true"

spark-class org.apache.spark.deploy.master.Master --host spark-master --port 7077 --webui-port 8080
