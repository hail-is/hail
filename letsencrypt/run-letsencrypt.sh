#!/bin/bash
set -ex

# start service
kubectl -n default apply -f service.yaml
# https://hail.zulipchat.com/#narrow/stream/300487-Hail-Batch-Dev/topic/azure.20deploy.20error/near/286124602
# You cannot mix the create verb with a resourceName. Therefore the only options are:
#
# 1. the letsencrypt pod has permission to create secrets with any name
#
# 2. the letsencrypt-config secret already exists
#
# We choose option two. We create an empty secret if it does not exist.
kubectl create secret generic letsencrypt-config || k get secret letsencrypt-config

# stop existing letsencrypt pod
kubectl -n default delete pod --ignore-not-found=true letsencrypt
N=
while [[ $N != 0 ]]; do
    sleep 5
    N=$(kubectl -n default get pod --ignore-not-found=true --no-headers letsencrypt | wc -l | tr -d '[:space:]')
    echo N=$N
done

# run letsencrypt pod
kubectl -n default create -f $1
echo "Waiting for letsencrypt to complete..."
EC=""
while [[ $EC = "" ]]; do
    sleep 5
    EC=$(kubectl -n default get pod -o "jsonpath={.status.containerStatuses[0].state.terminated.exitCode}" letsencrypt)
    echo EC=$EC
done
kubectl -n default logs letsencrypt
if [[ $EC != 0 ]]; then
    exit $EC
fi

# cleanup
kubectl -n default delete pod --ignore-not-found=true letsencrypt
