#! /bin/sh

if [ $# -ne 4 ]
  then
    echo "Usage ./make-secret.sh <user> <password> <db> <host>"
    exit
fi

if [ -z "$(kubectl get secret get-users --ignore-not-found)" ]
then
    kubectl create secret generic get-users --from-literal=user="$1" --from-literal=password="$2" --from-literal=db="$3" --from-literal=host="$4"
else
    echo "'get-users' already exists. To delete type 'kubectl delete secret get-users'"
fi