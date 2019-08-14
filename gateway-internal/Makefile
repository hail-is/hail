.PHONY: build push deploy

PROJECT = $(shell gcloud config get-value project)
GATEWAY_INTERNAL_LATEST = gcr.io/$(PROJECT)/gateway-internal:latest
GATEWAY_INTERNAL_IMAGE = gcr.io/$(PROJECT)/gateway-internal:$(shell docker images -q --no-trunc gateway-internal | sed -e 's,[^:]*:,,')
DOMAIN ?= hail.is
INTERNAL_IP ?= 10.128.0.57

build:
	docker pull ubuntu:18.04
	-docker pull $(GATEWAY_INTERNAL_LATEST)
	docker build -t gateway-internal --cache-from gateway-internal,$(GATEWAY_INTERNAL_LATEST),ubuntu:18.04 .

push: build
	docker tag gateway-internal $(GATEWAY_INTERNAL_LATEST)
	docker push $(GATEWAY_INTERNAL_LATEST)
	docker tag gateway-internal $(GATEWAY_INTERNAL_IMAGE)
	docker push $(GATEWAY_INTERNAL_IMAGE)

deploy: push
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":true,"global":{"internal_ip":"$(INTERNAL_IP)"}}' service.yaml service.yaml.out
	kubectl -n default apply -f service.yaml.out
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":true,"gateway_internal_image":{"image":"$(GATEWAY_INTERNAL_IMAGE)"}}' deployment.yaml deployment.yaml.out
	kubectl -n default apply -f deployment.yaml.out
