ZONE := us-central1-a
PROJECT := $(shell gcloud config get-value project)
DOMAIN ?= hail.is
IP ?= 35.224.105.117
KUBERNETES_SERVER_URL ?= https://104.198.230.143

PROJECT := $(shell gcloud config get-value project)

CI_UTILS_LATEST = gcr.io/$(PROJECT)/ci-utils:latest
CI_UTILS_IMAGE = gcr.io/$(PROJECT)/ci-utils:$(shell docker images -q --no-trunc ci-utils | sed -e 's,[^:]*:,,')

CI_LATEST = gcr.io/$(PROJECT)/ci:latest
CI_IMAGE = gcr.io/$(PROJECT)/ci:$(shell docker images -q --no-trunc ci | sed -e 's,[^:]*:,,')

PYTHONPATH := $${PYTHONPATH:+$${PYTHONPATH}:}
PYTHON := PYTHONPATH=$(PYTHONPATH)../batch:../hail/python:../gear:../web_common python3

.PHONY: check
check:
	$(PYTHON) -m flake8 ci
	$(PYTHON) -m pylint --rcfile ../pylintrc ci --score=n

.PHONY: build-ci-utils
build-ci-utils:
	make -C ../docker build
	-docker pull $(CI_UTILS_LATEST)
	python3 jinja2_render.py '{"service_base_image":{"image":"service-base"}}' Dockerfile.ci-utils Dockerfile.ci-utils.out
	docker build -t ci-utils -f Dockerfile.ci-utils.out --cache-from ci-utils,$(CI_UTILS_LATEST),base .

.PHONY: push-ci-utils
push-ci-utils: build-ci-utils
	docker tag ci-utils $(CI_UTILS_LATEST)
	docker push $(CI_UTILS_LATEST)
	docker tag ci-utils $(CI_UTILS_IMAGE)
	docker push $(CI_UTILS_IMAGE)

.PHONY: build
build:
	make -C ../docker build
	-docker	pull $(CI_LATEST)
	python3 jinja2_render.py '{"service_base_image":{"image":"service-base"}}' Dockerfile Dockerfile.out
	docker build -t ci -f Dockerfile.out --cache-from ci,$(CI_LATEST),service-base ..

.PHONY: push
push: build
	docker tag ci $(CI_LATEST)
	docker push $(CI_LATEST)
	docker tag ci $(CI_IMAGE)
	docker push $(CI_IMAGE)

.PHONY: deploy
deploy: push push-ci-utils
	python3 jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":true,"default_ns":{"name":"default"},"batch_pods_ns":{"name":"batch-pods"},"ci_image":{"image":"$(CI_IMAGE)"},"global":{"project":"$(PROJECT)","zone":"$(ZONE)","ip":"$(IP)","domain":"$(DOMAIN)", "k8s_server_url":"$(KUBERNETES_SERVER_URL)"},"ci_utils_image":{"image":"$(CI_UTILS_IMAGE)"},"ci_database":{"user_secret_name":"sql-ci-user-config"}}' deployment.yaml deployment.yaml.out
	kubectl -n default apply -f deployment.yaml.out
