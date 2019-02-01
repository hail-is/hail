.PHONY: all build push deploy transform

PROJECT = $(shell gcloud config get-value project)
WEB_LOCAL=hail-web
WEB_TAG=gcr.io/$(PROJECT)/web:$(WEB_LOCAL)

WEB_SECRETS=hail-web-secrets

build:
		-kubectl delete secret $(WEB_SECRETS)
		kubectl create secret generic $(WEB_SECRETS) --from-env-file=.env
		docker build -t $(WEB_LOCAL) ./ --no-cache

push:
	docker tag $(WEB_LOCAL) $(WEB_TAG)
	docker push $(WEB_TAG)

transform:
	cat deployment.yaml.in \
	| sed -e "s,@image@,${WEB_TAG}," \
	| sed -e "s,@secrets@,${WEB_SECRETS}," > deployment.yaml

deploy:
	kubectl -n default apply -f deployment.yaml

all: build push transform deploy
