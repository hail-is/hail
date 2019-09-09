PROJECT = $(shell gcloud config get-value project)
DOMAIN ?= hail.is

ROUTER_RESOLVER_LATEST = gcr.io/$(PROJECT)/router-resolver:latest
ROUTER_RESOLVER_IMAGE = gcr.io/$(PROJECT)/router-resolver:$(shell docker images -q --no-trunc router-resolver:latest | sed -e 's,[^:]*:,,')

PYTHONPATH := $${PYTHONPATH:+$${PYTHONPATH}:}../hail/python
PYTHON := PYTHONPATH=$(PYTHONPATH) python3

PY_FILES := $(shell find router_resolver -iname \*.py -not -exec git check-ignore -q {} \; -print)

flake8-stmp: $(PY_FILES)
	$(PYTHON) -m flake8 router_resolver
	touch $@

pylint-stmp: $(PY_FILES)
	$(PYTHON) -m pylint --rcfile ../pylintrc router_resolver --score=n
	touch $@

.PHONY: check
check: flake8-stmp pylint-stmp

.PHONY: clean
clean:
	rm -f flake8-stmp pylint-stmp

.PHONY: build
build:
	make -C ../docker build
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
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":true,"router_resolver_image":{"image":"$(ROUTER_RESOLVER_IMAGE)"},"global":{"domain":"$(DOMAIN)"}}' deployment.yaml deployment.yaml.out
	kubectl -n default apply -f deployment.yaml.out
