#! /bin/sh

if [ $# -ne 4 ]
  then
    echo "Usage ./make-secret.sh <user> <password> <db> <host>"
    exit
fi

dbName="create-users"

if [ -z "$(kubectl get secret $dbName --ignore-not-found)" ]
then
    kubectl create secret generic $dbName --from-literal=user="$1" --from-literal=password="$2" --from-literal=db="$3" --from-literal=host="$4"
else
    echo "'$dbName' already exists. To delete type 'kubectl delete secret $dbName'"
fi