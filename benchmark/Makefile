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
	cd python/ && HAIL_BENCHMARK_VERSION=$(HAIL_VERSION) $(HAIL_PYTHON3) setup.py bdist_wheel

.PHONY: install
install: wheel
	-$(PIP) uninstall -y benchmark_hail
	$(PIP) install $(BENCHMARK_WHEEL)

BENCHMARK_DOCKER_TAG := benchmark_$(shell whoami)
PROJECT = $(shell gcloud config get-value project)
BENCHMARK_LATEST = gcr.io/$(PROJECT)/$(BENCHMARK_DOCKER_TAG):latest

ifndef HAIL_WHEEL
.PHONY: build-benchmark-image
build-benchmark-image:
	@echo the build-benchmark-image target requires the HAIL_WHEEL variable to be set.
	exit 1
else
.PHONY: build-benchmark-image
build-benchmark-image: wheel
	cp $(HAIL_WHEEL) .
	cp $(BENCHMARK_WHEEL) .
	# it's possible that the HAIL_WHEEL installs different dependencies, but this generally creates less work for docker
	cp ../hail/python/requirements.txt .
	docker build -t $(BENCHMARK_DOCKER_TAG) . --build-arg HAIL_WHEEL=$(notdir $(HAIL_WHEEL)) --build-arg BENCHMARK_WHEEL=$(notdir $(BENCHMARK_WHEEL))
	rm $(notdir $(HAIL_WHEEL))
	rm $(notdir $(BENCHMARK_WHEEL))
	rm requirements.txt
endif

.PHONY: push-benchmark-image
push-benchmark-image: build-benchmark-image
	docker tag $(BENCHMARK_DOCKER_TAG) $(BENCHMARK_LATEST)
	docker push $(BENCHMARK_LATEST)

BENCHMARK_ITERS ?= 3
BENCHMARK_REPLICATES ?= 5
HAIL_WHEEL_DESCRIPTOR ?= $(HAIL_VERSION)-$(SHORT_REVISION)
.PHONY: benchmark
benchmark: push-benchmark-image install
	$(HAIL_PYTHON3) scripts/benchmark_in_pipeline.py $(BENCHMARK_LATEST) gs://hail-benchmarks/$(shell whoami) $(HAIL_WHEEL_DESCRIPTOR) $(BENCHMARK_REPLICATES) $(BENCHMARK_ITERS)

clean:
	rm -rf python/dist/*
	rm -rf python/build/*
	rm -f $(notdir $(HAIL_WHEEL))
	rm -f $(notdir $(BENCHMARK_WHEEL))
	rm -f requirements.txt

