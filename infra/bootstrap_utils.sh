#!/bin/bash

set -e

if [[ -z "$HAIL" ]]; then
    echo 1>&2 "Path to local clone of hail repository must be set."
    exit 1
fi

source $HAIL/devbin/functions.sh

copy_images() {
    make -C $HAIL/docker/third-party copy

    make -C $HAIL/hail python/hail/hail_pip_version
    make -C $HAIL/docker/hailgenetics mirror-dockerhub-images
}

generate_ssl_certs() {
    mkdir -p ~/certs
    cd ~/certs
    openssl req -new -x509 \
            -subj /CN=hail-root \
            -nodes \
            -newkey rsa:4096 \
            -keyout hail-root-key.pem \
            -out hail-root-cert.pem \
            -days 365 \
            -sha256

    kubectl create secret generic \
            -n default ssl-config-hail-root \
            --from-file=hail-root-key.pem \
            --from-file=hail-root-cert.pem \
            --save-config \
            --dry-run=client \
            -o yaml \
        | kubectl apply -f -

    PYTHONPATH=$HAIL/hail/python \
            python3 $HAIL/tls/create_certs.py \
            default \
            $HAIL/tls/config.yaml \
            hail-root-key.pem \
            hail-root-cert.pem
    cd -
}

deploy_unmanaged() {
    make -C $HAIL/hail python/hailtop/hail_version

    copy_images
    generate_ssl_certs

    export NAMESPACE=default
    kubectl -n default apply -f $HAIL/ci/bootstrap.yaml
    make -C $HAIL pushed-private-ci-utils-image pushed-private-hail-buildkit-image
    make -C $HAIL pushed-private-batch-worker-image
    make -C $HAIL/internal-gateway envoy-xds-config deploy
    make -C $HAIL/bootstrap-gateway deploy
    make -C $HAIL/letsencrypt run
}

bootstrap() {
    if [ -z "$1" ] || [ -z "$2" ]; then
        echo "Usage: bootstrap <REPO_ORG>/hail:<BRANCH> <DEPLOY_STEP>"
        exit 1
    fi
    HAIL_BRANCH=$1
    DEPLOY_STEP=$2

    cd $HAIL
    export DOCKER_PREFIX=$(get_global_config_field docker_prefix)
    export BATCH_WORKER_IMAGE=$DOCKER_PREFIX/batch-worker:cache
    export HAIL_CI_UTILS_IMAGE=$DOCKER_PREFIX/ci-utils:cache
    export HAIL_BUILDKIT_IMAGE=$DOCKER_PREFIX/hail-buildkit:cache
    export HAIL_DEFAULT_NAMESPACE=$(get_global_config_field default_namespace)
    export HAIL_CI_STORAGE_URI=dummy
    export HAIL_CI_GITHUB_CONTEXT=dummy
    export PYTHONPATH=$HAIL/ci:$HAIL/batch:$HAIL/hail/python:$HAIL/gear

    if [ -n "$3" ] && [ -n "$4" ]; then
        extra_code_config="--extra-code-config {\"username\":\"""$3""\",\"login_id\":\"""$4""\"}"
    else
        extra_code_config=""
    fi
    python3 ci/bootstrap.py $extra_code_config $HAIL_BRANCH $(git rev-parse HEAD) $DEPLOY_STEP
    cd -
}
