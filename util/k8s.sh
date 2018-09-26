#!/bin/sh

hailcipod () {
    set -ex
    PODS=$(kubectl get pods -l app=hail-ci --no-headers)
    [[ $(echo $PODS | wc -l) -eq 1 ]] || exit -1
    echo $PODS | awk '{print $1}'
}

jobpod () {
    set -ex
    PODS=$(kubectl get pods --no-headers | grep -Ee "job-$1-.*(Running|Pending)")
    [[ $(echo $PODS | wc -l) -eq 1 ]] || exit -1
    echo $PODS | awk '{print $1}'
}

cilogs () {
    set -ex
    kubectl logs $(hailcipod) $@
}

joblogs () {
    set -ex
    POD=$(jobpod $1)
    shift
    kubectl logs $POD $@
}
