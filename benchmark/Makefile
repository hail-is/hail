include ../config.mk

ifndef HAIL_WHEEL
$(error HAIL_WHEEL is not set!)
endif

ifndef HAIL_PIP_VERSION
$(error HAIL_PIP_VERSION is not set!)
endif

SHORT_REVISION := $(shell git rev-parse --short=12 HEAD)

HAIL_PYTHON3 ?= python3
PIP := $(HAIL_PYTHON3) -m pip

BENCHMARK_WHEEL := python/dist/benchmark_hail-$(HAIL_PIP_VERSION)-py3-none-any.whl
$(BENCHMARK_WHEEL):
	cd python/ && HAIL_BENCHMARK_VERSION=$(HAIL_PIP_VERSION) $(HAIL_PYTHON3) setup.py -q bdist_wheel

.PHONY: wheel
wheel: $(BENCHMARK_WHEEL)

.PHONY: install
install: wheel
	-$(PIP) uninstall -y benchmark_hail
	$(PIP) -q install $(BENCHMARK_WHEEL)

.PHONY: cleanup_image
cleanup_image:
	-docker image rm `cat image_sha`
	rm -f image_sha
	-docker image rm `cat image_sha`
	rm -f pushed_image

BENCHMARK_DOCKER_TAG ?= $(shell whoami)
BECNHMARK_IMAGE_REPOSITORY ?= us-docker.pkg.dev/broad-ctsa/hail-benchmarks
BENCHMARK_REPO_BASE = $(BECNHMARK_IMAGE_REPOSITORY)/$(BENCHMARK_DOCKER_TAG)

.PHONY: hail-ubuntu-image
hail-ubuntu-image:
	$(MAKE) -C ../ $@

image_sha: hail-ubuntu-image wheel cleanup_image
	docker build --platform linux/amd64 -f Dockerfile -t $(BENCHMARK_DOCKER_TAG) ../ \
		--build-arg HAIL_UBUNTU=$(shell cat ../hail-ubuntu-image) \
		--build-arg BENCHMARK_WHEEL=benchmark/$(BENCHMARK_WHEEL) \
		--build-arg HAIL_WHEEL=$(HAIL_WHEEL)
	@printf $$(docker images -q --no-trunc $(BENCHMARK_DOCKER_TAG) | sed -e 's,[^:]*:,,') > image_sha
	@echo Image sha is `cat image_sha`

pushed_image: image_sha
	@printf $(BENCHMARK_REPO_BASE):`cat image_sha` > pushed_image
	@echo Pushed image is `cat pushed_image`
	docker tag $(BENCHMARK_DOCKER_TAG) `cat pushed_image`
	docker push `cat pushed_image`

BENCHMARK_ITERS ?= 3
BENCHMARK_REPLICATES ?= 5
HAIL_WHEEL_DESCRIPTOR ?= $(HAIL_PIP_VERSION)-$(SHORT_REVISION)
BENCHMARK_BUCKET ?= gs://hail-benchmarks-2
.PHONY: benchmark
benchmark: pushed_image install
	@echo Using pushed image `cat pushed_image`
	$(HAIL_PYTHON3) scripts/benchmark_in_batch.py `cat pushed_image` $(BENCHMARK_BUCKET)/$(shell whoami) $(HAIL_WHEEL_DESCRIPTOR) $(BENCHMARK_REPLICATES) $(BENCHMARK_ITERS)

clean: cleanup_image
	rm -rf python/dist/*
	rm -rf python/build/*
	rm -f $(notdir $(HAIL_WHEEL))
	rm -f $(notdir $(BENCHMARK_WHEEL))
	rm -f requirements.txt
