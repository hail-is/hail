.PHONY: all build push deploy transform

PROJECT = $(shell gcloud config get-value project)
LOCAL=hail-web
TAG=gcr.io/$(PROJECT)/web:$(LOCAL)
SHA = $(shell git rev-parse HEAD)
SECRETS = hail-web-secrets

build:
		-kubectl delete secret $(SECRETS)
		kubectl create secret generic $(SECRETS) --from-env-file=.env
		docker build -t $(LOCAL) ./ --no-cache

push:
	docker tag $(LOCAL) $(TAG)
	docker push $(TAG)


transform:
	cat deployment.yaml.in \
	| sed -e "s,@image@,${TAG}," \
	| sed -e "s,@secrets@,${SECRETS}," \
	| sed -e "s,@sha@,${SHA}," > deployment.yaml

deploy:
	kubectl -n default apply -f deployment.yaml

all: build push transform deploy
