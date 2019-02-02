.PHONY: all build push deploy transform

PROJECT = $(shell gcloud config get-value project)
LOCAL = hail-auth-gateway
TAG = gcr.io/$(PROJECT)/auth-gatweay:$(LOCAL)

SECRETS = hail-auth-gateway-secrets

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
	| sed -e "s,@secrets@,${SECRETS}," > deployment.yaml

deploy:
	kubectl -n default apply -f deployment.yaml

all: build push transform deploy
