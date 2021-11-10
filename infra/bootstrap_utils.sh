#!/bin/bash

export HAIL="$HOME/hail"

get_global_config_field() {
    kubectl get secret global-config --template={{.data.$1}} | base64 --decode
}

render_config_mk() {
    DOCKER_PREFIX=$(get_global_config_field docker_prefix)
    INTERNAL_IP=$(get_global_config_field internal_ip)
    IP=$(get_global_config_field ip)
    DOMAIN=$(get_global_config_field domain)
    cat >$HAIL/config.mk <<EOF
DOCKER_PREFIX := $DOCKER_PREFIX
INTERNAL_IP := $INTERNAL_IP
IP := $IP
DOMAIN := $DOMAIN

ifeq (\$(NAMESPACE),default)
SCOPE = deploy
DEPLOY = true
else
SCOPE = dev
DEPLOY = false
endif
EOF
}

copy_images() {
    cd $HAIL/docker/third-party
    DOCKER_PREFIX=$(get_global_config_field docker_prefix)
    DOCKER_PREFIX=$DOCKER_PREFIX ./copy_images.sh
    cd -
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

    render_config_mk
    copy_images
    generate_ssl_certs

    kubectl -n default apply -f $HAIL/ci/bootstrap.yaml
    make -C $HAIL/ci build-ci-utils build-hail-buildkit
    make -C $HAIL/batch build-worker
    make -C $HAIL/internal-gateway deploy
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
        extra_code_config="--extra-code-config {\"username\":\"""$3""\",\"email\":\"""$4""\"}"
    else
        extra_code_config=""
    fi
    python3 ci/bootstrap.py $extra_code_config $HAIL_BRANCH $(git rev-parse HEAD) $DEPLOY_STEP
    cd -
}
