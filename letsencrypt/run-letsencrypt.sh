#!/bin/bash
set -ex

# start service
kubectl apply -f service.yaml

# stop existing letsencrypt pod
kubectl delete pod --ignore-not-found=true letsencrypt
N=
while [[ $N != 0 ]]; do
    sleep 5
    N=$(kubectl get pod --ignore-not-found=true --no-headers letsencrypt | wc -l | tr -d '[:space:]')
    echo N=$N
done

# run letsencrypt pod
kubectl create -f letsencrypt-pod.yaml
echo "Waiting for letsencrypt to complete..."
EC=""
while [[ $EC = "" ]]; do
    sleep 5
    EC=$(kubectl get pod -o "jsonpath={.status.containerStatuses[0].state.terminated.exitCode}" letsencrypt)
    echo EC=$EC
done
kubectl logs letsencrypt
if [[ $EC != 0 ]]; then
    exit $EC
fi

# cleanup
kubectl delete pod --ignore-not-found=true letsencrypt
