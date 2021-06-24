include ../config.mk

PYTHONPATH := $${PYTHONPATH:+$${PYTHONPATH}:}../hail/python:../gear
PYTHON := PYTHONPATH=$(PYTHONPATH) python3

TOKEN = $(shell cat /dev/urandom | LC_ALL=C tr -dc 'a-z0-9' | head -c 12)

MEMORY_IMAGE := $(DOCKER_PREFIX)/memory:$(TOKEN)

.PHONY: check
check:
	$(PYTHON) -m flake8  --config ../setup.cfg memory
	$(PYTHON) -m pylint --rcfile ../pylintrc memory --score=n

.PHONY: build
build:
	$(MAKE) -C ../docker service-base
	python3 ../ci/jinja2_render.py '{"service_base_image":{"image":"'$$(cat ../docker/service-base-image-ref)'"}}' Dockerfile Dockerfile.out
	../docker-build.sh .. memory/Dockerfile.out $(MEMORY_IMAGE)

.PHONY: deploy
deploy: build
	! [ -z $(NAMESPACE) ]  # call this like: make deploy NAMESPACE=default
	kubectl -n $(NAMESPACE) apply -f service-account.yaml
	python3 ../ci/jinja2_render.py '{"default_ns":{"name":"$(NAMESPACE)"}}' service-account-batch-pods.yaml service-account-batch-pods.yaml.out
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":$(DEPLOY),"memory_image":{"image":"$(MEMORY_IMAGE)"},"global":{"docker_prefix":"$(DOCKER_PREFIX)","domain":"$(DOMAIN)"}}' deployment.yaml deployment.yaml.out
	kubectl -n $(NAMESPACE) apply -f deployment.yaml.out
