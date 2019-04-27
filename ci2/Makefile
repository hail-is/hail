.PHONY: build

PROJECT := $(shell gcloud config get-value project)
DOMAIN ?= hail.is
IP ?= 35.224.105.117

PYTHONPATH := $${PYTHONPATH:+$${PYTHONPATH}:}
PYTHON := PYTHONPATH=$(PYTHONPATH)../batch python3

PY_FILES := $(shell find ci -iname \*.py -not -exec git check-ignore -q {} \; -print)

flake8-stmp: $(PY_FILES)
	python3 -m flake8 ci
	touch $@

pylint-stmp: $(PY_FILES)
	$(PYTHON) -m pylint --rcfile ../pylintrc ci --score=n
	touch $@

check: flake8-stmp pylint-stmp

install-batch-locally:
	rm -rf batch
	mkdir batch
	cp -a ../batch/setup.py ../batch/MANIFEST.in ../batch/batch batch

ifeq ($(IN_HAIL_CI),1)
.PHONY: push deploy

PROJECT := $(shell gcloud config get-value project)

CI_UTILS_LATEST = gcr.io/$(PROJECT)/ci-utils:latest
CI_UTILS_IMAGE = gcr.io/$(PROJECT)/ci-utils:$(shell docker images -q --no-trunc ci-utils | sed -e 's,[^:]*:,,')

CI2_LATEST = gcr.io/$(PROJECT)/ci2:latest
CI2_IMAGE = gcr.io/$(PROJECT)/ci2:$(shell docker images -q --no-trunc ci2 | sed -e 's,[^:]*:,,')

build-ci-utils:
	make -C ../docker build
	-docker pull $(CI_UTILS_LATEST)
	docker build -t ci-utils -f Dockerfile.ci-utils --cache-from ci-utils,$(CI_UTILS_LATEST),base .

push-ci-utils: build-ci-utils
	docker tag ci-utils $(CI_UTILS_LATEST)
	docker push $(CI_UTILS_LATEST)
	docker tag ci-utils $(CI_UTILS_IMAGE)
	docker push $(CI_UTILS_IMAGE)

build: install-batch-locally
	make -C ../docker build
	-docker	pull $(CI2_LATEST)
	docker build -t ci2 -f Dockerfile.ci2 --cache-from ci2,$(CI2_LATEST),base,ubuntu:18.04 .

push: build
	docker tag ci2 $(CI2_LATEST)
	docker push $(CI2_LATEST)
	docker tag ci2 $(CI2_IMAGE)
	docker push $(CI2_IMAGE)

deploy: push push-ci-utils
	sed -e "s,@sha@,$(shell git rev-parse --short=12 HEAD)," \
	  -e "s,@image@,$(CI2_IMAGE)," \
	  -e "s,@ci_utils_image@,$(CI_UTILS_IMAGE)," \
	  -e "s,@gcp_project@,$(PROJECT)," \
	  -e "s,@ip@,$(IP)," \
	  -e "s,@domain@,$(DOMAIN)," \
	  < deployment.yaml.in > deployment.yaml
	kubectl -n default apply -f deployment.yaml
else
build-ci-utils:
	make -C ../docker build
	docker build -t ci-utils -f Dockerfile.ci-utils .

build: install-batch-locally
	make -C ../docker build
	docker build . -t ci2 -f Dockerfile.ci2
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

# Paradox: can't push locally, but ci2 needs ci-utils.  Just use the latest.
run-local: PROXY_IP=$(shell gcloud compute instances list --filter=name=dk-test --format="value(EXTERNAL_IP)")
run-local: restart-proxy
	HAIL_GCP_PROJECT=$(PROJECT) \
	  HAIL_DOMAIN=$(DOMAIN) \
	  HAIL_IP=$(IP) \
	  HAIL_CI_UTILS_IMAGE=gcr.io/$(PROJECT)/ci-utils:latest \
	  $(PYTHON) -c 'import ci; ci.run()'

run-docker:
	docker run -p 5000:5000 ci2
