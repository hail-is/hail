.PHONY: all build push deploy transform

PROJECT = $(shell gcloud config get-value project)
LOCAL = hail-web
TAG = gcr.io/$(PROJECT)/hail-web:$(shell docker images -q --no-trunc ${LOCAL} | awk -F':' '{print $$2}')
SHA = $(shell git rev-parse HEAD)
SECRETS = hail-web-secrets

build:
		-kubectl delete secret $(SECRETS)
		kubectl create secret generic $(SECRETS) --from-env-file=.env
		docker rmi --force $(LOCAL)
		docker build -t $(LOCAL) ./
push:
	docker rmi --force $(TAG)
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
