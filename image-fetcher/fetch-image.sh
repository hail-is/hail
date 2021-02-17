#!/bin/sh
set -e

docker_registry=${DOCKER_PREFIX%%/*}

gcloud_auth() {
    gcloud -q auth activate-service-account --key-file=/secrets/gcr-pull.json && \
        gcloud -q auth configure-docker "${docker_registry}"
}

if ! gcloud_auth > gcloud-auth.log 2>&1; then
    1>&2 cat gcloud-auth.log
    exit 1
else
    cat gcloud-auth.log
fi

DEFAULT_NAMESPACE=$(jq -r '.default_namespace' < /deploy-config/deploy-config.json)
case $DEFAULT_NAMESPACE in
    default)
	NOTEBOOK_BASE_PATH=''
	;;
    *)
	NOTEBOOK_BASE_PATH="/$DEFAULT_NAMESPACE/notebook"
	;;
esac

ln -s /ssl-config/ssl-config.curlrc "$HOME/.curlrc"

echo "Namespace: $DEFAULT_NAMESPACE; Home: $HOME"
while true; do
    if curl -sSL "https://notebook$NOTEBOOK_BASE_PATH/images" > image-fetch-output.log 2>&1;
    then
        for image in "$DOCKER_PREFIX/base:latest" \
                         gcr.io/google.com/cloudsdktool/cloud-sdk:310.0.0-alpine \
                         $(cat image-fetch-output.log); do
            docker pull "$image" || true
        done
        sleep 360
    else
        1>&2 cat image-fetch-output.log
    fi
done
