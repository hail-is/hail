.PHONY: build clean

PROJECT := $(shell gcloud config get-value project)

PYTHONPATH := $${PYTHONPATH:+$${PYTHONPATH}:}
PYTHON := PYTHONPATH=$(PYTHONPATH)../batch:../hailjwt python3

PY_FILES := $(shell find builder -iname \*.py -not -exec git check-ignore -q {} \; -print)

flake8-stmp: $(PY_FILES)
	python3 -m flake8 builder
	touch $@

pylint-stmp: $(PY_FILES)
	$(PYTHON) -m pylint --rcfile ../pylintrc builder --score=n
	touch $@

check: flake8-stmp pylint-stmp

ifeq ($(IN_HAIL_CI),1)
.PHONY: push

BUILDER_UTILS_LATEST ?= gcr.io/$(PROJECT)/builder-utils:latest
BUILDER_UTILS_IMAGE ?= gcr.io/$(PROJECT)/builder-utils:$(shell docker images -q --no-trunc builder-utils | sed -e 's,[^:]*:,,')

build:
	make -C ../docker build
	-docker pull $(BUILDER_UTILS_LATEST)
	python3 jinja2_render.py '{"base_image":{"image":"base"}}' Dockerfile Dockerfile.out
	docker build -t builder-utils -f Dockerfile.out --cache-from builder,$(BUILDER_UTILS_LATEST),base .

push: build
	docker tag builder-utils $(BUILDER_UTILS_LATEST)
	docker push $(BUILDER_UTILS_LATEST)
	docker tag builder-utils $(BUILDER_UTILS_IMAGE)
	docker push $(BUILDER_UTILS_IMAGE)
else
build:
	make -C ../docker build
	python3 jinja2_render.py '{"base_image":{"image":"base"}}' Dockerfile Dockerfile.out
	docker build -t builder-utils -f Dockerfile.out .
endif

clean:
	rm -f flake8-stmp pylint-stmp
