include ../config.mk
include ../hail/version.mk

SHORT_REVISION := $(shell git rev-parse --short=12 HEAD)

HAIL_PYTHON3 ?= python3
PIP := $(HAIL_PYTHON3) -m pip

HAIL_BENCHMARK_VERSION := python/version
$(HAIL_BENCHMARK_VERSION):
	echo $(HAIL_PIP_VERSION) > $@

BENCHMARK_WHEEL := python/dist/benchmark_hail-$(HAIL_PIP_VERSION)-py3-none-any.whl
$(BENCHMARK_WHEEL):	$(shell git ls-files python)
	HAIL_BENCHMARK_VERSION=$(cat $(HAIL_BENCHMARK_VERSION)) && \
		cd python/ && $(HAIL_PYTHON3) setup.py -q bdist_wheel

.PHONY: wheel
wheel: $(BENCHMARK_WHEEL)

.PHONY: install
install:
	HAIL_BENCHMARK_VERSION=$(HAIL_PIP_VERSION) $(PIP) install python/

.PHONY: cleanup_image
cleanup_image:
	if [ -f image ]; then docker image rm `cat image`; fi
	rm -f image pushed_image

BENCHMARK_DOCKER_TAG ?= $(shell whoami)
BECNHMARK_IMAGE_REPOSITORY ?= us-docker.pkg.dev/broad-ctsa/hail-benchmarks
BENCHMARK_REPO_BASE = $(BECNHMARK_IMAGE_REPOSITORY)/$(BENCHMARK_DOCKER_TAG)

init = $(shell echo $(1) | sed -r 's|^/?[^/]*/?||')

.PHONY: hail-ubuntu-image
hail-ubuntu-image:
	$(MAKE) -C ../ $@

HAIL_WHEEL := ../hail/build/deploy/dist/hail-$(HAIL_PIP_VERSION)-py3-none-any.whl
$(HAIL_WHEEL): FORCE
	$(MAKE) -C ../hail wheel

image: hail-ubuntu-image $(HAIL_WHEEL) $(HAIL_BENCHMARK_VERSION) cleanup_image
	docker build --platform linux/amd64 -f Dockerfile -t $(BENCHMARK_DOCKER_TAG) ../ \
		--build-arg HAIL_UBUNTU=$(shell cat ../hail-ubuntu-image) \
		--build-arg HAIL_WHEEL_DIR=$(dir $(call init,$(HAIL_WHEEL)))
	@printf $$(docker images -q --no-trunc $(BENCHMARK_DOCKER_TAG) | sed -e 's,[^:]*:,,') > image
	@echo Image sha is `cat image`

pushed_image: image
	@printf $(BENCHMARK_REPO_BASE):`cat image` > pushed_image
	@echo Pushed image is `cat pushed_image`
	docker tag $(BENCHMARK_DOCKER_TAG) `cat pushed_image`
	docker push `cat pushed_image`

BENCHMARK_ITERS ?= 3
BENCHMARK_REPLICATES ?= 5
HAIL_WHEEL_DESCRIPTOR ?= $(HAIL_PIP_VERSION)-$(SHORT_REVISION)
BENCHMARK_BUCKET ?= gs://hail-benchmarks-2
.PHONY: submit
submit: pushed_image install
	@echo Using pushed image `cat pushed_image`
	$(HAIL_PYTHON3) scripts/benchmark_in_batch.py \
		`cat pushed_image` \
		$(BENCHMARK_BUCKET)/$(shell whoami) \
		$(HAIL_WHEEL_DESCRIPTOR) \
		$(BENCHMARK_REPLICATES) \
		$(BENCHMARK_ITERS)

clean: cleanup_image
	rm -rf python/dist/*
	rm -rf python/build/*
	rm -r $(HAIL_BENCHMARK_VERSION)

FORCE:
