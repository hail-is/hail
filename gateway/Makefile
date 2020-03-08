.PHONY: build push deploy

PROJECT = $(shell gcloud config get-value project)
GATEWAY_LATEST = gcr.io/$(PROJECT)/gateway:latest
GATEWAY_IMAGE = gcr.io/$(PROJECT)/gateway:$(shell docker images -q --no-trunc gateway | sed -e 's,[^:]*:,,')
DOMAIN ?= hail.is
IP ?= 35.188.91.25

build:
	docker pull ubuntu:18.04
	-docker pull $(GATEWAY_LATEST)
	docker build -t gateway --cache-from gateway,$(GATEWAY_LATEST),ubuntu:18.04 .

push: build
	docker tag gateway $(GATEWAY_LATEST)
	docker push $(GATEWAY_LATEST)
	docker tag gateway $(GATEWAY_IMAGE)
	docker push $(GATEWAY_IMAGE)

deploy: push
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":true,"global":{"ip":"$(IP)"}}' service.yaml service.yaml.out
	kubectl -n default apply -f service.yaml.out
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":true,"gateway_image":{"image":"$(GATEWAY_IMAGE)"}}' deployment.yaml deployment.yaml.out
	kubectl -n default apply -f deployment.yaml.out
