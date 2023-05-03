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
.PHONY: wheel
wheel:
	cd python/ && HAIL_BENCHMARK_VERSION=$(HAIL_PIP_VERSION) $(HAIL_PYTHON3) setup.py -q bdist_wheel

.PHONY: install
install: wheel
	-$(PIP) uninstall -y benchmark_hail
	$(PIP) -q install $(BENCHMARK_WHEEL)

.PHONY: cleanup_image
cleanup_image:
	rm -f image_sha
	rm -f pushed_image

BENCHMARK_PROJECT ?= hail-vdc
BENCHMARK_DOCKER_TAG ?= benchmark_$(shell whoami)
BENCHMARK_REPO_BASE = us-docker.pkg.dev/$(BENCHMARK_PROJECT)/$(BENCHMARK_DOCKER_TAG)
DOCKER_ROOT_IMAGE := ubuntu:20.04

ifndef HAIL_WHEEL
image_sha:
	@echo the image_sha target requires the HAIL_WHEEL variable to be set.
	exit 1
else
image_sha: wheel cleanup_image
	cp $(HAIL_WHEEL) .
	cp $(BENCHMARK_WHEEL) .
	# it's possible that the HAIL_WHEEL installs different dependencies, but this generally creates less work for docker
	cp ../hail/python/requirements.txt .
	cp ../hail/python/hailtop/requirements.txt hailtop_requirements.txt
	python3 ../ci/jinja2_render.py '{"global":{"docker_root_image":"$(DOCKER_ROOT_IMAGE)"}}' Dockerfile Dockerfile.out
	docker build --platform linux/amd64 -f Dockerfile.out -t $(BENCHMARK_DOCKER_TAG) . --build-arg HAIL_WHEEL=$(notdir $(HAIL_WHEEL)) --build-arg BENCHMARK_WHEEL=$(notdir $(BENCHMARK_WHEEL))
	@printf $$(docker images -q --no-trunc $(BENCHMARK_DOCKER_TAG) | sed -e 's,[^:]*:,,') > image_sha
	@echo Image sha is `cat image_sha`
	rm $(notdir $(HAIL_WHEEL))
	rm $(notdir $(BENCHMARK_WHEEL))
	rm requirements.txt
endif

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

clean:
	rm -rf python/dist/*
	rm -rf python/build/*
	rm -f $(notdir $(HAIL_WHEEL))
	rm -f $(notdir $(BENCHMARK_WHEEL))
	rm -f requirements.txt
	rm -f hailtop_requirements.txt
