.PHONY: echo-ip gcloud-config build-out create-deployment build-out-k8s create-k8s-secrets create-k8s-certs-volume deploy-k8s \
	delete-deployment tear-down create-address

export PROJECT ?= hail-vdc-staging
export REGION ?= us-central1
export ZONE ?= us-central1-a
export DOMAIN ?= staging.hail.is
export IP ?= $(shell gcloud compute addresses describe --region $(REGION) site --format='value(address)')

build-out: create-artifacts-bucket create-deployment build-out-k8s

echo-ip:
	echo IP=$(IP)

gcloud-config:
	gcloud config set project $(PROJECT)
	gcloud config set compute/region $(REGION)
	gcloud config set compute/zone $(ZONE)

# deployment manager cannot manage appspot.com bucket because it requires domain ownership verification
create-artifacts-bucket:
	docker pull alpine:3.8
	docker tag alpine:3.8 gcr.io/hail-vdc-staging/alpine:3.8
	docker push gcr.io/$(PROJECT)/alpine:3.8

create-deployment: gcloud-config
	sed -e 's,@project@,$(PROJECT),g' \
	  -e 's,@region@,$(REGION),g' \
	  -e 's,@zone@,$(ZONE),g' \
	  < gcp-config.yaml.in > gcp-config.yaml
	gcloud beta -q deployment-manager deployments create default --config gcp-config.yaml
	gcloud container clusters get-credentials vdc
# FIXME this was insufficient, had to add cseed@broadinstitute.org as admin
	kubectl create clusterrolebinding project-editor-cluster-admin-binding --clusterrole cluster-admin --user projectEditor:$(PROJECT)

build-out-k8s: k8s-config create-k8s-secrets run-letsencrypt deploy-k8s

k8s-config:
	kubectl apply -f k8s-config.yaml

create-k8s-secrets:
	kubectl apply -f secrets.yaml
	kubectl delete secrets  --ignore-not-found=true gcr-push-service-account-key gcr-pull-key letsencrypt-config
	gcloud iam service-accounts keys create \
	  --iam-account=gcr-push@$(PROJECT).iam.gserviceaccount.com \
	  gcr-push-service-account-key.json
	kubectl create secret generic gcr-push-service-account-key --from-file=gcr-push-service-account-key.json
	rm -f gcr-push-service-account-key.json
	gcloud iam service-accounts keys create \
	  --iam-account=gcr-pull@$(PROJECT).iam.gserviceaccount.com \
	  gcr-pull.json
	kubectl create secret generic gcr-pull-key --from-file=gcr-pull.json
	rm -f gcr-pull.json
# empty secret to be filled in by letsencrypt
	kubectl create secret generic letsencrypt-config

run-letsencrypt:
	$(MAKE) -C ../letsencrypt start-service
	$(MAKE) -C ../gateway LETSENCRYPT_ONLY=1 deploy
	kubectl -n default rollout status -w deployment gateway-deployment
	$(MAKE) -C ../letsencrypt run

deploy-k8s:
	$(MAKE) -C ../batch deploy
# FIXME skip ci until it can handle running as a secondary
#	$(MAKE) -C ../ci run-service
	$(MAKE) -C ../notebook deploy
	$(MAKE) -C ../image-fetcher deploy
	$(MAKE) -C ../scorecard deploy
	$(MAKE) -C ../site deploy
	$(MAKE) -C ../upload deploy
# last so the services are up
	$(MAKE) -C ../gateway deploy

delete-artifacts-bucket:
# don't fail if doesn't exist
	-gsutil -m rm -r gs://artifacts.$(PROJECT).appspot.com

tear-down: delete-deployment delete-artifacts-bucket

delete-deployment: gcloud-config
	gcloud beta -q deployment-manager deployments delete default

create-address:  gcloud-config
	gcloud beta -q deployment-manager deployments create address --config gcp-address.yaml

delete-address:  gcloud-config
	gcloud beta -q deployment-manager deployments delete address
