#!/bin/bash
set -ex

unset SPARK_MASTER_HOST
unset SPARK_MASTER_PORT

export SPARK_DAEMON_JAVA_OPTS="-Djava.net.preferIPv4Stack=true"

/spark-2.2.0-bin-hadoop2.7/bin/spark-class org.apache.spark.deploy.worker.Worker -c 2 --webui-port 8081 spark://spark-master:7077
