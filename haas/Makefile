.PHONY: all build push deploy transform

PROJECT = $(shell gcloud config get-value project)
WEB_LOCAL=hail-web
API_LOCAL=hail-api-gateway

WEB_TAG=gcr.io/$(PROJECT)/web:$(WEB_LOCAL)
API_TAG=gcr.io/$(PROJECT)/web:$(API_LOCAL)

WEB_SECRETS=hail-web-secrets
API_SECRETS=hail-api-gateway-secrets

build:
		-kubectl delete secret $(WEB_SECRETS)
		-kubectl delete secret $(API_SECRETS)
		kubectl create secret generic $(WEB_SECRETS) --from-env-file=./packages/public/.env
		kubectl create secret generic $(API_SECRETS) --from-env-file=./packages/hail-api-gateway/.env
		docker build -t $(WEB_LOCAL) ./packages/public/ --no-cache
		docker build -t $(API_LOCAL) ./packages/hail-api-gateway/ --no-cache

push:
	docker tag $(WEB_LOCAL) $(WEB_TAG)
	docker tag $(API_LOCAL) $(API_TAG)
	docker push $(WEB_TAG)
	docker push $(API_TAG)

transform:
	cat deployment.yaml.in \
	| sed -e "s,@image_web@,${WEB_TAG}," \
	| sed -e "s,@image_api@,${API_TAG}," \
	| sed -e "s,@web_secrets@,${WEB_SECRETS}," \
	| sed -e "s,@api_secrets@,${API_SECRETS}," > deployment.yaml

deploy:
	kubectl -n default apply -f deployment.yaml

all: build push transform deploy
