include ../config.mk

TOKEN = $(shell cat /dev/urandom | LC_ALL=C tr -dc 'a-z0-9' | head -c 12)

ADDRESS_IMAGE := $(DOCKER_PREFIX)/address:$(TOKEN)

PYTHONPATH := $${PYTHONPATH:+$${PYTHONPATH}:}../hail/python:../gear:../web_common
PYTHON := PYTHONPATH=$(PYTHONPATH) python3

.PHONY: check
check:
	$(PYTHON) -m flake8  --config ../setup.cfg address
	$(PYTHON) -m pylint --rcfile ../pylintrc address --score=n
	curlylint .

.PHONY: build
build:
	$(MAKE) -C ../docker service-base
	python3 ../ci/jinja2_render.py '{"service_base_image":{"image":"'$$(cat ../docker/service-base)'"}}' Dockerfile Dockerfile.out
	../docker-build.sh .. address/Dockerfile.out $(ADDRESS_IMAGE)

.PHONY: deploy
deploy: build
	! [ -z $(NAMESPACE) ]  # call this like: make deploy NAMESPACE=default
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":$(DEPLOY),"address_image":{"image":"$(ADDRESS_IMAGE)"},"default_ns":{"name":"$(NAMESPACE)"},"global":{"project":"$(PROJECT)","domain":"$(DOMAIN)"}}' deployment.yaml deployment.yaml.out
	kubectl -n $(NAMESPACE) apply -f service-account.yaml
	kubectl -n $(NAMESPACE) apply -f deployment.yaml.out
