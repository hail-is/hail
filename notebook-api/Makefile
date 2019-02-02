.PHONY: all build push deploy transform

PROJECT = $(shell gcloud config get-value project)

NOTEBOOK_SVC_NAME:="notebook-api"
NOTEBOOK_IMAGE = gcr.io/$(PROJECT)/$(NOTEBOOK_SVC_NAME)

build:
	docker build -t $(NOTEBOOK_SVC_NAME) ./ --no-cache

push:
	docker tag $(NOTEBOOK_SVC_NAME) $(NOTEBOOK_IMAGE)
	docker push $(NOTEBOOK_IMAGE)

transform:
	cat deployment.yaml.in \
	| sed -e "s,@image@,${NOTEBOOK_IMAGE}," \
	| sed -e "s,@sha@,$(shell git rev-parse HEAD)," > deployment.yaml

deploy:
	kubectl -n default apply -f deployment.yaml

run-docker-local:
	docker run -p 5000:5000 -e BATCH_USE_KUBE_CONFIG=1 -v ~/:/root notebook-api

all: build push transform deploy
