.PHONY: build-out create-deployment build-out-k8s create-k8s-secrets create-k8s-certs-volume deploy-k8s \
	delete-deployment tear-down

export PROJECT ?= hail-vdc
export DOMAIN ?= staging.hail.is
export IP ?= 35.188.91.25

build-out: create-deployment build-out-k8s

create-deployment:
	gcloud config set project $(PROJECT)
	gcloud beta -q deployment-manager deployments create default --config config.yaml
# FIXME integrate into config
	gsutil iam ch serviceAccount:vdc-svc@hail-vdc.iam.gserviceaccount.com:objectViewer gs://artifacts.hail-vdc.appspot.com
	gcloud container clusters get-credentials vdc

build-out-k8s: create-k8s-secrets create-k8s-certs-volume run-letsencrypt deploy-k8s

create-k8s-secrets:
	kubectl apply -f secrets.yaml
	kubectl create secret generic upload-oauth2-client-secret \
	  --from-file=./client_secret.json

create-k8s-certs-volume:
	kubectl apply -f ../site/certs-volume.yaml

run-letsencrypt:
	$(MAKE) -C ../site push-run-letsencrypt run-letsencrypt

deploy-k8s:
	$(MAKE) -C ../batch push-batch deploy-batch
# FIXME skip ci until it can handle running as a secondary
	$(MAKE) -C ../ci run-service
	$(MAKE) -C ../scorecard push deploy
	$(MAKE) -C ../upload push deploy
	$(MAKE) -C ../site push-site deploy-site

delete-deployment:
	gcloud beta -q --project $(PROJECT) deployment-manager deployments delete default

tear-down:
# just to be safe
	gcloud config set project $(PROJECT)
# FIXME integrate into config
	gsutil iam ch -d serviceAccount:vdc-svc@hail-vdc.iam.gserviceaccount.com:objectViewer gs://artifacts.hail-vdc.appspot.com
	gcloud beta -q --project $(PROJECT) deployment-manager deployments delete default
	/bin/bash delete-gcr-images.sh
