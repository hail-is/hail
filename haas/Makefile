.PHONY: build push deploy transform

PROJECT = $(shell gcloud config get-value project)
WEB_LOCAL=hail-web
API_LOCAL=hail-api-gateway

WEB_TAG=gcr.io/$(PROJECT)/web:$(WEB_LOCAL)
API_TAG=gcr.io/$(PROJECT)/web:$(API_LOCAL)

build:
		kubectl delete secret hail-web-secrets
		kubectl delete secret hail-api-gateway-secrets
		kubectl create secret generic hail-web-secrets --from-env-file=./packages/public/.env
		kubectl create secret generic hail-api-gateway-secrets --from-env-file=./packages/hail-api-gateway/.env
		docker build -t $(WEB_LOCAL) ./packages/public/
		docker build -t $(API_LOCAL) ./packages/hail-api-gateway/

push:
	docker tag $(WEB_LOCAL) $(WEB_TAG)
	docker tag $(API_LOCAL) $(API_TAG)
	docker push $(WEB_TAG)
	docker push $(API_TAG)

transform:
	cat deployment.yaml.in | sed -e "s,@image_web@,${WEB_TAG}," \
	| sed -e "s,@image_api@,${API_TAG}," > deployment.yaml

deploy: build push transform
	kubectl -n default apply -f deployment.yaml
