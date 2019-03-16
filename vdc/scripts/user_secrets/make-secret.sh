#! /bin/sh

if [ $# -ne 4 ]
  then
    echo "Usage ./make-secret.sh <user> <password> <db> <host>"
    exit
fi

if [ -z "$(kubectl get secret user-secrets --ignore-not-found)" ]
then
    kubectl create secret generic user-secrets --from-literal=user="$1" --from-literal=password="$2" --from-literal=db="$3" --from-literal=host="$4"
else
    echo "'user-secrets' already exists. To delete type 'kubectl delete secret user-secrets'"
fi