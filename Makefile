.DEFAULT_GOAL := default

include config.mk

SERVICES := auth batch ci memory notebook monitoring website
SERVICES_IMAGES := $(patsubst %, %-image, $(SERVICES))
SERVICES_MODULES := $(SERVICES) gear web_common
CHECK_SERVICES_MODULES := $(patsubst %, check-%, $(SERVICES_MODULES))

HAILTOP_VERSION := hail/python/hailtop/hail_version
SERVICES_IMAGE_DEPS = hail-ubuntu-image $(HAILTOP_VERSION) $(shell git ls-files hail/python/hailtop gear web_common)

EMPTY :=
SPACE := $(EMPTY) $(EMPTY)
EXTRA_PYTHONPATH := hail/python:$(subst $(SPACE),:,$(SERVICES_MODULES))
ifeq ($(PYTHONPATH), "")
PYTHONPATH := $(EXTRA_PYTHONPATH)
else
PYTHONPATH := $(PYTHONPATH):$(EXTRA_PYTHONPATH)
endif
PYTHON := PYTHONPATH=$(PYTHONPATH) python3

default:
	@echo Do not use this makefile to build hail, for information on how to \
	     build hail see: https://hail.is/docs/0.2/
	@false

.PHONY: check-all
check-all: check-hail check-services

.PHONY: check-hail
check-hail:
	$(MAKE) -C hail/python check

.PHONY: check-services
check-services: $(CHECK_SERVICES_MODULES)

.PHONY: check-%
$(CHECK_SERVICES_MODULES): check-%:
	$(PYTHON) -m flake8  --config setup.cfg $*
	$(PYTHON) -m pylint --rcfile pylintrc --recursive=y $* --score=n
	$(PYTHON) -m mypy --config-file setup.cfg $*
	$(PYTHON) -m isort $* --check-only --diff
	$(PYTHON) -m black $* --line-length=120 --skip-string-normalization --check --diff
	curlylint $*
	cd $* && bash ../check-sql.sh

.PHONY: check-pip-requirements
check-pip-requirements:
	./check_pip_requirements.sh \
		hail/python/hailtop \
		hail/python \
		hail/python/dev \
		gear \
		web_common \
		auth \
		batch \
		ci \
		memory

.PHONY: install-dev-requirements
install-dev-requirements:
	python3 -m pip install \
		-r hail/python/pinned-requirements.txt \
		-r hail/python/dev/pinned-requirements.txt \
		-r gear/pinned-requirements.txt \
		-r web_common/pinned-requirements.txt \
		-r auth/pinned-requirements.txt \
		-r batch/pinned-requirements.txt \
		-r ci/pinned-requirements.txt \
		-r memory/pinned-requirements.txt \

hail/python/hailtop/pinned-requirements.txt: hail/python/hailtop/requirements.txt
	./generate-linux-pip-lockfile.sh hail/python/hailtop

hail/python/pinned-requirements.txt: hail/python/requirements.txt hail/python/hailtop/pinned-requirements.txt
	./generate-linux-pip-lockfile.sh hail/python

hail/python/dev/pinned-requirements.txt: hail/python/dev/requirements.txt hail/python/pinned-requirements.txt
	./generate-linux-pip-lockfile.sh hail/python/dev

gear/pinned-requirements.txt: hail/python/hailtop/pinned-requirements.txt gear/requirements.txt
	./generate-linux-pip-lockfile.sh gear

web_common/pinned-requirements.txt: gear/pinned-requirements.txt web_common/requirements.txt
	./generate-linux-pip-lockfile.sh web_common

auth/pinned-requirements.txt: web_common/pinned-requirements.txt auth/requirements.txt
	./generate-linux-pip-lockfile.sh auth

batch/pinned-requirements.txt: web_common/pinned-requirements.txt batch/requirements.txt
	./generate-linux-pip-lockfile.sh batch

ci/pinned-requirements.txt: web_common/pinned-requirements.txt ci/requirements.txt
	./generate-linux-pip-lockfile.sh ci

memory/pinned-requirements.txt: gear/pinned-requirements.txt memory/requirements.txt
	./generate-linux-pip-lockfile.sh memory

.PHONY: generate-pip-lockfiles
generate-pip-lockfiles: hail/python/hailtop/pinned-requirements.txt
generate-pip-lockfiles: hail/python/pinned-requirements.txt
generate-pip-lockfiles: hail/python/dev/pinned-requirements.txt
generate-pip-lockfiles: gear/pinned-requirements.txt
generate-pip-lockfiles: web_common/pinned-requirements.txt
generate-pip-lockfiles: auth/pinned-requirements.txt
generate-pip-lockfiles: batch/pinned-requirements.txt
generate-pip-lockfiles: ci/pinned-requirements.txt
generate-pip-lockfiles: memory/pinned-requirements.txt

$(HAILTOP_VERSION):
	$(MAKE) -C hail python/hailtop/hail_version

hail-ubuntu-image: docker/hail-ubuntu
	$(eval HAIL_UBUNTU_IMAGE := $(DOCKER_PREFIX)/hail-ubuntu:$(TOKEN))
	python3 ci/jinja2_render.py '{"global":{"docker_prefix":"$(DOCKER_PREFIX)"}}' docker/hail-ubuntu/Dockerfile docker/hail-ubuntu/Dockerfile.out
	./docker-build.sh docker/hail-ubuntu Dockerfile.out $(HAIL_UBUNTU_IMAGE)
	echo $(HAIL_UBUNTU_IMAGE) > $@

base-image: hail-ubuntu-image docker/Dockerfile.base
	$(eval BASE_IMAGE := $(DOCKER_PREFIX)/base:$(TOKEN))
	python3 ci/jinja2_render.py '{"hail_ubuntu_image":{"image":"'$$(cat hail-ubuntu-image)'"}}' docker/Dockerfile.base docker/Dockerfile.base.out
	./docker-build.sh . docker/Dockerfile.base.out $(BASE_IMAGE)
	echo $(BASE_IMAGE) > $@

private-repo-hailgenetics-hail-image: hail-ubuntu-image docker/hailgenetics/hail/Dockerfile $(shell git ls-files hail/src/main hail/python)
	$(eval PRIVATE_REPO_HAILGENETICS_HAIL_IMAGE := $(DOCKER_PREFIX)/hailgenetics/hail:$(TOKEN))
	$(MAKE) -C hail wheel
	cp hail/build/deploy/dist/hail-$$(cat hail/python/hail/hail_pip_version)-py3-none-any.whl .
	tar -cvf wheel-container.tar hail-$$(cat hail/python/hail/hail_pip_version)-py3-none-any.whl
	rm hail-$$(cat hail/python/hail/hail_pip_version)-py3-none-any.whl
	python3 ci/jinja2_render.py '{"hail_ubuntu_image":{"image":"'$$(cat hail-ubuntu-image)'"}}' docker/hailgenetics/hail/Dockerfile docker/hailgenetics/hail/Dockerfile.out
	./docker-build.sh . docker/hailgenetics/hail/Dockerfile.out $(PRIVATE_REPO_HAILGENETICS_HAIL_IMAGE)
	rm wheel-container.tar
	echo $(PRIVATE_REPO_HAILGENETICS_HAIL_IMAGE) > $@

.PHONY: docs
docs:
	$(MAKE) -C hail hail-docs-no-test batch-docs
	gcloud storage cp gs://hail-common/builds/0.1/docs/hail-0.1-docs-5a6778710097.tar.gz .
	mkdir -p hail/build/www/docs/0.1
	tar -xvf hail-0.1-docs-5a6778710097.tar.gz -C hail/build/www/docs/0.1 --strip-components 2
	rm hail-0.1-docs-5a6778710097.tar.gz
	tar czf docs.tar.gz -C hail/build/www .

website-image: docs

$(SERVICES_IMAGES): %-image: $(SERVICES_IMAGE_DEPS) $(shell git ls-files $$*)
	$(eval IMAGE := $(DOCKER_PREFIX)/$*:$(TOKEN))
	python3 ci/jinja2_render.py '{"hail_ubuntu_image":{"image":"'$$(cat hail-ubuntu-image)'"}}' $*/Dockerfile $*/Dockerfile.out
	./docker-build.sh . $*/Dockerfile.out $(IMAGE)
	echo $(IMAGE) > $@

ci-utils-image: hail-ubuntu-image $(SERVICES_IMAGE_DEPS) ci/jinja2_render.py ci/wait-for.py ci/create_database.py
	$(eval CI_UTILS_IMAGE := $(DOCKER_PREFIX)/ci-utils:$(TOKEN))
	python3 ci/jinja2_render.py '{"hail_ubuntu_image":{"image":"'$$(cat hail-ubuntu-image)'"}}' ci/Dockerfile.ci-utils ci/Dockerfile.ci-utils.out
	./docker-build.sh . ci/Dockerfile.ci-utils.out $(CI_UTILS_IMAGE)
	echo $(CI_UTILS_IMAGE) > $@

hail-buildkit-image: ci/buildkit/Dockerfile
	$(eval HAIL_BUILDKIT_IMAGE := $(DOCKER_PREFIX)/hail-buildkit:$(TOKEN))
	python3 ci/jinja2_render.py '{"global":{"docker_prefix":"$(DOCKER_PREFIX)"}}' ci/buildkit/Dockerfile ci/buildkit/Dockerfile.out
	./docker-build.sh ci buildkit/Dockerfile.out $(HAIL_BUILDKIT_IMAGE)
	echo $(HAIL_BUILDKIT_IMAGE) > $@

batch/jars/junixsocket-selftest-2.3.3-jar-with-dependencies.jar:
	mkdir -p batch/jars
	cd batch/jars && curl -LO https://github.com/kohlschutter/junixsocket/releases/download/junixsocket-parent-2.3.3/junixsocket-selftest-2.3.3-jar-with-dependencies.jar

batch/src/main/java/is/hail/JVMEntryway.class: batch/src/main/java/is/hail/JVMEntryway.java batch/jars/junixsocket-selftest-2.3.3-jar-with-dependencies.jar
	javac -cp batch/jars/junixsocket-selftest-2.3.3-jar-with-dependencies.jar $<

batch-worker-image: batch/src/main/java/is/hail/JVMEntryway.class $(SERVICES_IMAGE_DEPS) $(shell git ls-files batch)
	$(eval BATCH_WORKER_IMAGE := $(DOCKER_PREFIX)/batch-worker:$(TOKEN))
	python3 ci/jinja2_render.py '{"hail_ubuntu_image":{"image":"'$$(cat hail-ubuntu-image)'"},"global":{"cloud":"$(CLOUD)"}}' batch/Dockerfile.worker batch/Dockerfile.worker.out
	./docker-build.sh . batch/Dockerfile.worker.out $(BATCH_WORKER_IMAGE)
	echo $(BATCH_WORKER_IMAGE) > $@
