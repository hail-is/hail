#!/bin/bash

set -ex

dir=$(mktemp -d)

echo logs in $dir

retry() {
    "$@" ||
        (sleep 2 && "$@") ||
        (sleep 5 && "$@");
}

sync_and_restart_pod() {
    pod=$1
    echo restarting $pod
    retry krsync.sh -av --progress --stats \
              --exclude='*.log' --exclude='.mypy_cache' --exclude='__pycache__' --exclude='*~' \
              $(pwd)/web_common/web_common \
              $(pwd)/gear/gear \
              $(pwd)/hail/python/hailtop \
              $(pwd)/$name/$name \
              ${pod}@${namespace}:/usr/local/lib/python3.7/dist-packages/ >> $dir/.dksync-$pod.log
    retry kubectl exec $pod -n $namespace -- kill -2 1 >> $dir/.dksync-$pod.log
    date >> $dir/.dksync-$pod.log
    echo done $pod
}

name=$1
namespace=$2

kubectl get pods \
        -l app=$name \
        --namespace $namespace \
        -o json \
    | jq '.items[] | .metadata.name' -r \
         >$dir/pods.json

while read -r line
do
    for pod in $(cat $dir/pods.json)
    do
        sync_and_restart_pod $pod &
    done
    wait
done < <(fswatch -o -l 2 --exclude='*.log' --exclude='.mypy_cache' --exclude='__pycache__' --exclude='*~' $(pwd))
