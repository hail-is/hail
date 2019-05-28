.PHONY: build push deploy

PROJECT = $(shell gcloud config get-value project)
ROUTER_RESOLVER_LATEST = gcr.io/$(PROJECT)/router-resolver:latest
ROUTER_RESOLVER_IMAGE = gcr.io/$(PROJECT)/router-resolver:$(shell docker images -q --no-trunc router-resolver:latest | sed -e 's,[^:]*:,,')

build:
	make -C ../docker build
	-docker pull $(ROUTER_RESOLVER_LATEST)
	python3 jinja2_render.py '{"base_image":{"image":"base"}}' Dockerfile Dockerfile.out
	docker build -f Dockerfile.out -t router-resolver --cache-from router-resolver,$(ROUTER_RESOLVER_LATEST),base .

push: build
	docker tag router-resolver $(ROUTER_RESOLVER_LATEST)
	docker push $(ROUTER_RESOLVER_LATEST)
	docker tag router-resolver $(ROUTER_RESOLVER_IMAGE)
	docker push $(ROUTER_RESOLVER_IMAGE)

deploy: push
	python3 jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":true,"router_resolver_image":{"image":"$(ROUTER_RESOLVER_IMAGE)"}}' deployment.yaml deployment.yaml.out
	kubectl -n default apply -f deployment.yaml.out
