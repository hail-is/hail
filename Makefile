.DEFAULT_GOAL := default

include config.mk

SERVICES := auth batch ci monitoring website
SERVICES_PLUS_ADMIN_POD := $(SERVICES) admin-pod
SERVICES_IMAGES := $(patsubst %, %-image, $(SERVICES_PLUS_ADMIN_POD))
SERVICES_DATABASES := $(patsubst %, %-db, $(SERVICES))
SERVICES_MODULES := $(SERVICES) gear web_common
CHECK_SERVICES_MODULES := $(patsubst %, check-%, $(SERVICES_MODULES))
SPECIAL_IMAGES := hail-ubuntu batch-worker letsencrypt

HAILGENETICS_IMAGES = $(foreach img,hail vep-grch37-85 vep-grch38-95,hailgenetics-$(img))
CI_IMAGES = ci-utils hail-buildkit base hail-run
PRIVATE_REGISTRY_IMAGES = $(patsubst %, pushed-private-%-image, $(SPECIAL_IMAGES) $(SERVICES_PLUS_ADMIN_POD) $(CI_IMAGES) $(HAILGENETICS_IMAGES))

HAILTOP_VERSION := hail/python/hailtop/version.py
SERVICES_IMAGE_DEPS = hail-ubuntu-image $(HAILTOP_VERSION) $(shell git ls-files hail/python/hailtop gear web_common)

EMPTY :=
SPACE := $(EMPTY) $(EMPTY)
EXTRA_PYTHONPATH := hail/python:$(subst $(SPACE),:,$(SERVICES_MODULES))

PYTHONPATH ?= ""
ifeq ($(PYTHONPATH), "")
PYTHONPATH := $(EXTRA_PYTHONPATH)
else
PYTHONPATH := $(PYTHONPATH):$(EXTRA_PYTHONPATH)
endif
PYTHON := PYTHONPATH="$(PYTHONPATH)" python3

# mill integration

MILL := sh mill
MILLOPTS ?=

default:
	@echo Do not use this makefile to build hail, for information on how to \
	     build hail see: https://hail.is/docs/0.2/
	@false

.PHONY: check-all
check-all: check-hail check-services

.PHONY: check-hail-fast
check-hail-fast:
	ruff check hail
	ruff format hail --diff
	$(PYTHON) -m pyright hail/python/hailtop

	ruff check hail/python/test/hailtop/batch
	$(PYTHON) -m pyright hail/python/test/hailtop/batch

.PHONY: pylint-hailtop
pylint-hailtop:
	# pylint on hail is still a work in progress
	$(PYTHON) -m pylint --rcfile pylintrc hail/python/hailtop --score=n

.PHONY: check-hail
check-hail: check-hail-fast pylint-hailtop
	cd hail && HAIL_BUILD_MODE=CI SCALA_VERSION=2.13 $(MILL) $(MILLOPTS) hail.__.checkFormat + hail.__.fix --check

.PHONY: check-batch
check-batch: check-batch-fast pylint-batch
	cd batch/jvm-entryway && $(MILL) $(MILLOPTS) checkFormat + fix --check

.PHONY: check-services
check-services: $(CHECK_SERVICES_MODULES)

.PHONY: pylint-%
pylint-%:
	$(PYTHON) -m pylint --rcfile pylintrc --recursive=y $* --score=n

.PHONY: check-%-fast
check-%-fast:
	ruff check $*
	ruff format $* --diff
	$(PYTHON) -m pyright $*
	curlylint $*
	cd $* && bash ../check-sql.sh

.PHONY: check-%
$(CHECK_SERVICES_MODULES): check-%: check-%-fast pylint-%

.PHONY: isort-%
isort-%:
	ruff check --select I --fix $*


.PHONY: check-pip-requirements
check-pip-requirements:
	./check_pip_requirements.sh \
		hail/python/hailtop \
		hail/python \
		hail/python/dev \
		gear \
		web_common \
		batch \
		ci

.PHONY: install-dev-requirements
install-dev-requirements:
	python3 -m pip install \
		-r hail/python/pinned-requirements.txt \
		-r hail/python/dev/pinned-requirements.txt \
		-r gear/pinned-requirements.txt \
		-r web_common/pinned-requirements.txt \
		-r batch/pinned-requirements.txt \
		-r ci/pinned-requirements.txt

.PHONY: update-hail-ubuntu
update-hail-ubuntu:
	@command -v skopeo >/dev/null 2>&1 || { echo 'ERROR: skopeo is required. Install with: brew install skopeo'; exit 1; }
	python3 update-hail-ubuntu.py
	NAMESPACE=default $(MAKE) -C docker/third-party copy

.PHONY: update-gateways-envoy
update-gateways-envoy:
	python3 update-envoy.py
	$(MAKE) -C gateway deploy
	$(MAKE) -C internal-gateway deploy
	kubectl rollout status deployment gateway-deployment
	kubectl rollout status deployment internal-gateway
	kubectl get deployment gateway-deployment -o jsonpath='{.spec.template.spec.containers[0].image}'
	kubectl get deployment internal-gateway -o jsonpath='{.spec.template.spec.containers[0].image}'
	@echo ''
	@echo 'If traffic to hail.is services seems degraded, roll back with:'
	@echo '  kubectl rollout undo deployment gateway-deployment && kubectl rollout undo deployment internal-gateway'

.PHONY: generate-pip-lockfiles
generate-pip-lockfiles:
	./generate-pip-lockfile.sh hail/python/hailtop
	./generate-pip-lockfile.sh hail/python
	./generate-pip-lockfile.sh hail/python/dev
	./generate-pip-lockfile.sh gear
	./generate-pip-lockfile.sh web_common
	./generate-pip-lockfile.sh batch
	./generate-pip-lockfile.sh ci

$(HAILTOP_VERSION):
	$(MAKE) -C hail python-version-info


%-image: IMAGE_NAME = $(patsubst %-image,%,$@):$(TOKEN)
hailgenetics-%-image: IMAGE_NAME = hailgenetics/$(patsubst hailgenetics-%-image,%,$@):$(TOKEN)

hail-ubuntu-image: $(shell git ls-files docker/hail-ubuntu gear/pinned-requirements.txt)
	./docker-build.sh . docker/hail-ubuntu/Dockerfile $(IMAGE_NAME) --build-arg DOCKER_PREFIX=$(DOCKER_PREFIX) --build-arg SKIP_GCLOUD_PROFILER=false
	echo $(IMAGE_NAME) > $@

base-image: hail-ubuntu-image docker/Dockerfile.base
	./docker-build.sh . docker/Dockerfile.base $(IMAGE_NAME) --build-arg BASE_IMAGE=$(shell cat hail-ubuntu-image)
	echo $(IMAGE_NAME) > $@

hail-run-image: base-image hail/Dockerfile.hail-run hail/python/pinned-requirements.txt hail/python/dev/pinned-requirements.txt docker/core-site.xml
	$(MAKE) -C hail wheel
	./docker-build.sh . hail/Dockerfile.hail-run $(IMAGE_NAME) --build-arg BASE_IMAGE=$(shell cat base-image)
	echo $(IMAGE_NAME) > $@

hailgenetics-hail-image: hail-ubuntu-image docker/hailgenetics/hail/Dockerfile $(shell git ls-files hail/src/main hail/python)
	$(MAKE) HAIL_BUILD_MODE=Release -C hail wheel
	./docker-build.sh . docker/hailgenetics/hail/Dockerfile $(IMAGE_NAME) \
		--build-arg BASE_IMAGE=$(shell cat hail-ubuntu-image)
	echo $(IMAGE_NAME) > $@

hail-dev-image: hailgenetics-hail-image hail/python/dev/pinned-requirements.txt
	./docker-build.sh . docker/Dockerfile.hail-dev $(IMAGE_NAME) \
		--build-arg HAIL_IMAGE=$(shell cat $<)
	echo $(IMAGE_NAME) > $@

hail-0.1-docs-5a6778710097.tar.gz:
	gcloud storage cp gs://hail-common/builds/0.1/docs/$@ .

hail/build/www: hail-0.1-docs-5a6778710097.tar.gz $(shell git ls-files hail)
	@echo !!! This target does not render the notebooks because it takes a long time !!!
	$(MAKE) -C hail hail-docs-do-not-render-notebooks batch-docs
	mkdir -p hail/build/www/docs/0.1
	tar -xvf hail-0.1-docs-5a6778710097.tar.gz -C hail/build/www/docs/0.1 --strip-components 2
	touch $@  # Copying into the dir does not necessarily touch it

website/website/docs: hail/build/www
	cp -r hail/build/www/docs website/website/
	touch $@  # Copying into the dir does not necessarily touch it

docs.tar.gz: hail/build/www
	tar czf docs.tar.gz -C hail/build/www .

website-image: docs.tar.gz

.SECONDEXPANSION:
$(SERVICES_IMAGES): %-image: $(SERVICES_IMAGE_DEPS) $$(shell git ls-files $$* ':!:**/deployment.yaml')
	./docker-build.sh . $*/Dockerfile $(IMAGE_NAME) --build-arg BASE_IMAGE=$(shell cat hail-ubuntu-image)
	echo $(IMAGE_NAME) > $@

ci-utils-image: base-image $(SERVICES_IMAGE_DEPS) $(shell git ls-files ci)
	./docker-build.sh . ci/Dockerfile.ci-utils $(IMAGE_NAME) --build-arg BASE_IMAGE=$(shell cat base-image)
	echo $(IMAGE_NAME) > $@

hail-buildkit-image: ci/buildkit/Dockerfile
	./docker-build.sh ci buildkit/Dockerfile $(IMAGE_NAME) --build-arg DOCKER_PREFIX=$(DOCKER_PREFIX)
	echo $(IMAGE_NAME) > $@

services/ui/dist/.built: services/ui/node_modules/.package-lock.json $(shell git ls-files services/ui)
	cd services/ui && npm run build
	touch $@

ci/ci/static/compiled-js/flaky_tests.js: services/ui/dist/.built
	mkdir -p $(@D)
	cp services/ui/dist/ci/flaky_tests.js $@

ci-image: ci/ci/static/compiled-js/flaky_tests.js

batch/batch/front_end/static/compiled-js/job.js: services/ui/dist/.built
	mkdir -p $(@D)
	cp services/ui/dist/batch/job.js $@

batch-image: batch/batch/front_end/static/compiled-js/job.js

batch/batch/driver/static/compiled-js/index.js: services/ui/dist/.built
	mkdir -p $(@D)
	cp services/ui/dist/batch_driver/index.js $@

batch-image: batch/batch/driver/static/compiled-js/index.js

monitoring/monitoring/static/compiled-js/index.js: services/ui/dist/.built
	mkdir -p $(@D)
	cp services/ui/dist/monitoring/index.js $@

monitoring-image: monitoring/monitoring/static/compiled-js/index.js

auth/auth/static/compiled-js/index.js: services/ui/dist/.built
	mkdir -p $(@D)
	cp services/ui/dist/auth/index.js $@

auth-image: auth/auth/static/compiled-js/index.js

batch/jvm-entryway/out/assembly.dest/out.jar: $(shell git ls-files batch/jvm-entryway)
	cd batch/jvm-entryway && $(MILL) $(MILLOPTS) assembly

batch-worker-image: batch/jvm-entryway/out/assembly.dest/out.jar $(SERVICES_IMAGE_DEPS) $(shell git ls-files batch)
	python3 ci/jinja2_render.py '{"hail_ubuntu_image":{"image":"'$$(cat hail-ubuntu-image)'"},"global":{"cloud":"$(CLOUD)"}}' batch/Dockerfile.worker batch/Dockerfile.worker.out
	./docker-build.sh . batch/Dockerfile.worker.out $(IMAGE_NAME)
	echo $(IMAGE_NAME) > $@

hailgenetics-vep-grch37-85-image: hail-ubuntu-image
	./docker-build.sh docker/vep docker/vep/grch37/85/Dockerfile $(IMAGE_NAME) \
		--build-arg BASE_IMAGE=$(shell cat hail-ubuntu-image)
	echo $(IMAGE_NAME) > $@

hailgenetics-vep-grch38-95-image: hail-ubuntu-image
	./docker-build.sh docker/vep docker/vep/grch38/95/Dockerfile $(IMAGE_NAME) \
		--build-arg BASE_IMAGE=$(shell cat hail-ubuntu-image)
	echo $(IMAGE_NAME) > $@

terra-batch-image: batch-image
	./docker-build.sh . batch/terra-chart/Dockerfile.batch $(IMAGE_NAME) \
		--build-arg BASE_IMAGE=$(shell cat batch-image)
	echo $(IMAGE_NAME) > $@

terra-batch-worker-image: batch-worker-image
	$(MAKE) -C hail shadowJar HAIL_BUILD_MODE=Release
	./docker-build.sh . batch/terra-chart/Dockerfile.worker $(IMAGE_NAME) \
		--build-arg BASE_IMAGE=$(shell cat batch-worker-image)
	echo $(IMAGE_NAME) > $@

letsencrypt-image:
	./docker-build.sh letsencrypt Dockerfile $(IMAGE_NAME)
	echo $(IMAGE_NAME) > $@

$(PRIVATE_REGISTRY_IMAGES): pushed-private-%-image: %-image
	! [ -z $(NAMESPACE) ]  # call this like: make ... NAMESPACE=default
	[ $(DOCKER_PREFIX) != docker.io ]  # DOCKER_PREFIX should be an internal private registry
	! [ -z $(DOCKER_PREFIX) ]  # DOCKER_PREFIX must not be empty
	docker tag $(shell cat $*-image) $(DOCKER_PREFIX)/$(shell cat $*-image)
	docker push $(DOCKER_PREFIX)/$(shell cat $*-image)
	echo $(DOCKER_PREFIX)/$(shell cat $*-image) > $@

.PHONY: clean-image-targets
clean-image-targets:
	rm -f *-image

.PHONY: local-mysql
local-mysql:
	cd docker/mysql && docker compose up -d

.PHONY: $(SERVICES_DATABASES)
$(SERVICES_DATABASES): %-db: local-mysql
ifdef DROP
$(SERVICES_DATABASES): %-db:
	MYSQL_PWD=pw mysql -h 127.0.0.1 -u root -e 'DROP DATABASE `local-$*`'
else
$(SERVICES_DATABASES): %-db:
	python3 ci/create_local_database.py $* local-$*
endif

.PHONY: tailwind-compile-watch
tailwind-compile-watch:
	cd web_common && npx tailwindcss --watch -i input.css -o web_common/static/css/output.css

run-dev-proxy: ci/ci/static/compiled-js/flaky_tests.js \
    batch/batch/front_end/static/compiled-js/job.js \
    batch/batch/driver/static/compiled-js/index.js \
    monitoring/monitoring/static/compiled-js/index.js \
    auth/auth/static/compiled-js/index.js
DEVSERVER_TARGETS = tailwind-compile-watch run-dev-proxy ui-js-watch ui-js-watch-batch ui-js-watch-batch-driver ui-js-watch-monitoring ui-js-watch-auth

.PHONY: run-dev-proxy
run-dev-proxy:
	adev runserver --root . --static web_common/web_common/static devbin/dev_proxy.py

services/ui/node_modules/.package-lock.json: services/ui/package.json services/ui/package-lock.json
	npm ci --prefix services/ui

.PHONY: ui-js-watch
ui-js-watch: services/ui/node_modules/.package-lock.json
	cd services/ui && npx esbuild src/ci/flaky_tests.tsx --bundle --jsx=automatic --format=esm --outfile=../../ci/ci/static/compiled-js/flaky_tests.js --minify --watch=forever

.PHONY: ui-js-watch-batch
ui-js-watch-batch: services/ui/node_modules/.package-lock.json
	cd services/ui && npx esbuild src/batch/job.tsx --bundle --jsx=automatic --format=esm --outfile=../../batch/batch/front_end/static/compiled-js/job.js --minify --watch=forever

.PHONY: ui-js-watch-batch-driver
ui-js-watch-batch-driver: services/ui/node_modules/.package-lock.json
	cd services/ui && npx esbuild src/batch_driver/index.tsx --bundle --jsx=automatic --format=esm --outfile=../../batch/batch/driver/static/compiled-js/index.js --minify --watch=forever

.PHONY: ui-js-watch-monitoring
ui-js-watch-monitoring: services/ui/node_modules/.package-lock.json
	cd services/ui && npx esbuild src/monitoring/index.tsx --bundle --jsx=automatic --format=esm --outfile=../../monitoring/monitoring/static/compiled-js/index.js --minify --watch=forever

.PHONY: ui-js-watch-auth
ui-js-watch-auth: services/ui/node_modules/.package-lock.json
	cd services/ui && npx esbuild src/auth/index.tsx --bundle --jsx=automatic --format=esm --outfile=../../auth/auth/static/compiled-js/index.js --minify --watch=forever

.PHONY: check-devserver-deps
check-devserver-deps:
	@if ! command -v adev > /dev/null 2>&1; then \
		echo 'error: adev not found (aiohttp-devtools is not on PATH)' >&2; \
		echo 'fix: run "make install-dev-requirements" (and/or "source .venv/bin/activate" if using a venv)' >&2; \
		exit 1; \
	fi
	@python3 -c "\
import importlib.metadata as m, sys; \
pkgs = ['gear', 'web_common', 'batch', 'ci', 'monitoring', 'auth']; \
installed = {d.name for d in m.distributions()}; \
missing = [p for p in pkgs if p not in installed]; \
(print('error: missing packages: ' + ', '.join(missing), file=sys.stderr), \
 print('run: pip install ' + ' '.join('-e ' + p for p in missing), file=sys.stderr), \
 sys.exit(1)) if missing else None"

.PHONY: devserver
devserver: check-devserver-deps
	$(MAKE) -j 7 $(DEVSERVER_TARGETS)

.PHONY: benchmark
benchmark: hail-dev-image
	$(MAKE) -C hail HAIL_DEV_IMAGE="$(shell cat $<)" benchmark
