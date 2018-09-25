#!/bin/bash
set -ex

# start service
kubectl apply -f service.yaml

# stop existing site deployment
kubectl delete --ignore-not-found=true -f site-deployment.yaml
N=
while [[ $N != 0 ]]; do
    sleep 5
    N=$(kubectl get pods -l app=site --ignore-not-found=true --no-headers | wc -l | tr -d '[:space:]')
    echo N=$N
done

# stop existing run-letsencrypt pod
kubectl delete pod --ignore-not-found=true run-letsencrypt
N=
while [[ $N != 0 ]]; do
    sleep 5
    N=$(kubectl get pod --ignore-not-found=true --no-headers run-letsencrypt | wc -l | tr -d '[:space:]')
    echo N=$N
done

# run run-letsencrypt pod
kubectl create -f run-letsencrypt-pod.yaml
echo "Waiting for run-letsencrypt to complete..."
EC=""
while [[ $EC = "" ]]; do
    sleep 5
    EC=$(kubectl get pod -o "jsonpath={.status.containerStatuses[0].state.terminated.exitCode}" run-letsencrypt)
    echo EC=$EC
done
kubectl logs run-letsencrypt
if [[ $EC != 0 ]]; then
    exit $EC
fi

# cleanup
kubectl delete pod --ignore-not-found=true run-letsencrypt
