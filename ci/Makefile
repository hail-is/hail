.PHONY: build base-image clean

PROJECT := $(shell gcloud config get-value project)
DOMAIN ?= hail.is
IP ?= 35.224.105.117

PYTHONPATH := $${PYTHONPATH:+$${PYTHONPATH}:}
PYTHON := PYTHONPATH=$(PYTHONPATH)../batch:../hail/python python3

PY_FILES := $(shell find ci -iname \*.py -not -exec git check-ignore -q {} \; -print)

flake8-stmp: $(PY_FILES)
	python3 -m flake8 ci
	touch $@

pylint-stmp: $(PY_FILES)
	$(PYTHON) -m pylint --rcfile ../pylintrc ci --score=n
	touch $@

check: flake8-stmp pylint-stmp

ifeq ($(IN_HAIL_CI),1)
.PHONY: push deploy

PROJECT := $(shell gcloud config get-value project)

CI_UTILS_LATEST = gcr.io/$(PROJECT)/ci-utils:latest
CI_UTILS_IMAGE = gcr.io/$(PROJECT)/ci-utils:$(shell docker images -q --no-trunc ci-utils | sed -e 's,[^:]*:,,')

CI_LATEST = gcr.io/$(PROJECT)/ci:latest
CI_IMAGE = gcr.io/$(PROJECT)/ci:$(shell docker images -q --no-trunc ci | sed -e 's,[^:]*:,,')

build-ci-utils:
	make -C ../docker build
	-docker pull $(CI_UTILS_LATEST)
	python3 jinja2_render.py '{"base_image":{"image":"base"}}' Dockerfile.ci-utils Dockerfile.ci-utils.out
	docker build -t ci-utils -f Dockerfile.ci-utils.out --cache-from ci-utils,$(CI_UTILS_LATEST),base .

push-ci-utils: build-ci-utils
	docker tag ci-utils $(CI_UTILS_LATEST)
	docker push $(CI_UTILS_LATEST)
	docker tag ci-utils $(CI_UTILS_IMAGE)
	docker push $(CI_UTILS_IMAGE)

build:
	make -C ../docker build
	-docker	pull $(CI_LATEST)
	python3 jinja2_render.py '{"base_image":{"image":"base"}}' Dockerfile Dockerfile.out
	docker build -t ci -f Dockerfile.out --cache-from ci,$(CI_LATEST),base ..

push: build
	docker tag ci $(CI_LATEST)
	docker push $(CI_LATEST)
	docker tag ci $(CI_IMAGE)
	docker push $(CI_IMAGE)

deploy: push push-ci-utils
	python3 jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":true,"default_ns":{"name":"default"},"ci_image":{"image":"$(CI_IMAGE)"},"global":{"project":"$(PROJECT)","ip":"$(IP)","domain":"$(DOMAIN)"},"ci_utils_image":{"image":"$(CI_UTILS_IMAGE)"},"ci_database":{"user_secret_name":"sql-ci-ci-user-config"}}' deployment.yaml deployment.yaml.out
	kubectl -n default apply -f deployment.yaml.out
else
build-ci-utils:
	make -C ../docker build
	docker build -t ci-utils -f Dockerfile.ci-utils .

build:
	make -C ../docker build
	docker build -t ci -f Dockerfile ..
endif

PROXY_REMOTE_PORT = 3001

restart-proxy: ZONE = $(shell gcloud compute instances list --filter=name=dk-test --format="value(ZONE)")
restart-proxy:
	-kill $(shell cat proxy.pid)
	-kill -9 $(shell cat proxy.pid)
	-rm -rf proxy.pid
	$(shell gcloud compute \
	  --project "$(PROJECT)" \
	  ssh \
	  --zone "$(ZONE)" \
	  "dk-test" \
	  --ssh-flag="-R 0.0.0.0:${PROXY_REMOTE_PORT}:127.0.0.1:5000" \
	  --ssh-flag='-N' \
	  --ssh-flag='-T' \
	  --ssh-flag='-v' \
	  --dry-run) > proxy.log 2>proxy.err & echo $$! > proxy.pid
	sleep 2 && kill -0 $$(cat proxy.pid)

# Paradox: can't push locally, but ci needs ci-utils.  Just use the latest.
run-local: PROXY_IP=$(shell gcloud compute instances list --filter=name=dk-test --format="value(EXTERNAL_IP)")
run-local: restart-proxy
	HAIL_GCP_PROJECT=$(PROJECT) \
	  HAIL_DOMAIN=$(DOMAIN) \
	  HAIL_IP=$(IP) \
	  HAIL_CI_UTILS_IMAGE=gcr.io/hail-vdc/ci-utils:f477fdaa98d0e640f1bc2003ccfe1d2a86cf2d7531ff595476ca2412d28fc678 \
	  $(PYTHON) -c 'import ci; ci.run()'

run-docker:
	docker run -p 5000:5000 ci

clean:
	rm -f flake8-stmp pylint-stmp
