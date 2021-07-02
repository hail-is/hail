include ../config.mk

PYTHONPATH := $${PYTHONPATH:+$${PYTHONPATH}:}../hail/python:../gear
PYTHON := PYTHONPATH=$(PYTHONPATH) python3

MEMORY_LATEST = $(DOCKER_PREFIX)/memory:latest
MEMORY_IMAGE = $(DOCKER_PREFIX)/memory:$(shell docker images -q --no-trunc memory | sed -e 's,[^:]*:,,')

.PHONY: check
check:
	$(PYTHON) -m flake8  --config ../setup.cfg memory
	$(PYTHON) -m pylint --rcfile ../pylintrc memory --score=n

.PHONY: build
build:
	$(MAKE) -C ../docker build
	-docker pull $(MEMORY_LATEST)
	python3 ../ci/jinja2_render.py '{"service_base_image":{"image":"service-base"}}' Dockerfile Dockerfile.out
	docker build -t memory -f Dockerfile.out --cache-from memory,$(MEMORY_LATEST),service-base ..

.PHONY: push
push: build
	docker tag memory $(MEMORY_LATEST)
	docker push $(MEMORY_LATEST)
	docker tag memory $(MEMORY_IMAGE)
	docker push $(MEMORY_IMAGE)

.PHONY: deploy

deploy: push
	! [ -z $(NAMESPACE) ]  # call this like: make deploy NAMESPACE=default
	kubectl -n $(NAMESPACE) apply -f service-account.yaml
	python3 ../ci/jinja2_render.py '{"default_ns":{"name":"$(NAMESPACE)"}}' service-account-batch-pods.yaml service-account-batch-pods.yaml.out
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":$(DEPLOY),"memory_image":{"image":"$(MEMORY_IMAGE)"},"global":{"docker_prefix":"$(DOCKER_PREFIX)","domain":"$(DOMAIN)"}}' deployment.yaml deployment.yaml.out
	kubectl -n $(NAMESPACE) apply -f deployment.yaml.out
