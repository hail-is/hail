include ../config.mk

ROUTER_RESOLVER_LATEST = $(DOCKER_PREFIX)/router-resolver:latest
ROUTER_RESOLVER_IMAGE = $(DOCKER_PREFIX)/router-resolver:$(shell docker images -q --no-trunc router-resolver:latest | sed -e 's,[^:]*:,,')

EXTRA_PYTHONPATH := ../hail/python:../gear
PYTHON := PYTHONPATH=$${PYTHONPATH:+$${PYTHONPATH}:}$(EXTRA_PYTHONPATH) python3

.PHONY: check
check:
	$(PYTHON) -m flake8 router_resolver
	$(PYTHON) -m pylint --rcfile ../pylintrc router_resolver --score=n

.PHONY: build
build:
	$(MAKE) -C ../docker build
	-docker pull $(ROUTER_RESOLVER_LATEST)
	python3 ../ci/jinja2_render.py '{"service_base_image":{"image":"service-base"}}' Dockerfile Dockerfile.out
	docker build -f Dockerfile.out -t router-resolver --cache-from router-resolver,$(ROUTER_RESOLVER_LATEST),service-base ..

.PHONY: push
push: build
	docker tag router-resolver $(ROUTER_RESOLVER_LATEST)
	docker push $(ROUTER_RESOLVER_LATEST)
	docker tag router-resolver $(ROUTER_RESOLVER_IMAGE)
	docker push $(ROUTER_RESOLVER_IMAGE)

.PHONY: deploy
deploy: push
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":$(DEPLOY),"router_resolver_image":{"image":"$(ROUTER_RESOLVER_IMAGE)"},"global":{"domain":"$(DOMAIN)"}}' deployment.yaml deployment.yaml.out
	kubectl -n default apply -f deployment.yaml.out
