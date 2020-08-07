ifndef HAIL_WHEEL
$(error HAIL_WHEEL is not set!)
endif

ifndef HAIL_VERSION
$(error HAIL_VERSION is not set!)
endif

SHORT_REVISION := $(shell git rev-parse --short=12 HEAD)

HAIL_PYTHON3 ?= python3
PIP := $(HAIL_PYTHON3) -m pip

BENCHMARK_WHEEL := python/dist/benchmark_hail-$(HAIL_VERSION)-py3-none-any.whl
.PHONY: wheel
wheel:
	cd python/ && HAIL_BENCHMARK_VERSION=$(HAIL_VERSION) $(HAIL_PYTHON3) setup.py -q bdist_wheel

.PHONY: install
install: wheel
	-$(PIP) uninstall -y benchmark_hail
	$(PIP) -q install $(BENCHMARK_WHEEL)

.PHONY: cleanup_image
cleanup_image:
	rm -f image_sha
	rm -f pushed_image

BENCHMARK_DOCKER_TAG := benchmark_$(shell whoami)
PROJECT = $(shell gcloud config get-value project)
BENCHMARK_REPO_BASE = gcr.io/$(PROJECT)/$(BENCHMARK_DOCKER_TAG)

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
	docker build -t $(BENCHMARK_DOCKER_TAG) . --build-arg HAIL_WHEEL=$(notdir $(HAIL_WHEEL)) --build-arg BENCHMARK_WHEEL=$(notdir $(BENCHMARK_WHEEL))
	@printf $(shell docker images -q --no-trunc $(BENCHMARK_DOCKER_TAG) | sed -e 's,[^:]*:,,') > image_sha
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
HAIL_WHEEL_DESCRIPTOR ?= $(HAIL_VERSION)-$(SHORT_REVISION)
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

