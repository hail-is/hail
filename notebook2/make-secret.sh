#!/bin/sh

if [ $# -ne 4 ]
  then
    echo "Usage ./make-secret.sh <user> <password> <db> <host>"
    exit
fi

kubectl create secret generic get-users --from-literal=user="$1" --from-literal=password="$2" --from-literal=db="$3" --from-literal=host="$4"