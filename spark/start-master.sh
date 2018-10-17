#!/bin/bash
set -x

unset SPARK_MASTER_HOST
unset SPARK_MASTER_PORT

# echo "$POD_IP spark-master" >> /etc/hosts
echo "0.0.0.0 spark-master" >> /etc/hosts

export SPARK_DAEMON_JAVA_OPTS="-Djava.net.preferIPv4Stack=true"

/spark-2.2.0-bin-hadoop2.7/bin/spark-class org.apache.spark.deploy.master.Master --host spark-master --port 7077 --webui-port 8080
