#!/bin/sh
set -ex

gcloud -q auth activate-service-account \
  --key-file=/secrets/gcr-pull.json

gcloud -q auth configure-docker

DEFAULT_NAMESPACE=$(jq -r '.default_namespace' < /deploy-config/deploy-config.json)
case $DEFAULT_NAMESPACE in
    default)
	NOTEBOOK_BASE_PATH=''
	;;
    *)
	NOTEBOOK_BASE_PATH="/$DEFAULT_NAMESPACE/notebook"
	;;
esac

while true; do
    for image in "gcr.io/$PROJECT/base:latest" google/cloud-sdk:237.0.0-alpine $(curl -sSL http://notebook$NOTEBOOK_BASE_PATH/images); do
    	docker pull $image
    done
    sleep 360
done
