#!/bin/bash
set -ex

unset SPARK_MASTER_HOST
unset SPARK_MASTER_PORT

export SPARK_DAEMON_JAVA_OPTS="-Djava.net.preferIPv4Stack=true"

spark-class org.apache.spark.deploy.worker.Worker -c 2 -p 9000 --webui-port 8081 spark://spark-master:7077
