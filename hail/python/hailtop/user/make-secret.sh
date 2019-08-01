#! /bin/sh

if [ $# -ne 5 ]
  then
    echo "Usage ./make-secret.sh <user> <password> <db> <host> <cloud-sql-instance-connection-name>"
    exit
fi

dbName="create-users"

kubectl create secret generic $dbName --from-literal=user="$1" --from-literal=password="$2" --from-literal=db="$3" --from-literal=host="$4" --from-literal=instance="$5"