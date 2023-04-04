include ../config.mk

PYTHONPATH := $${PYTHONPATH:+$${PYTHONPATH}:}../hail/python:../gear
PYTHON := PYTHONPATH=$(PYTHONPATH) python3
BLACK := $(PYTHON) -m black . --line-length=120 --skip-string-normalization

MEMORY_IMAGE := $(DOCKER_PREFIX)/memory:$(TOKEN)

.PHONY: check
check:
	$(PYTHON) -m flake8  --config ../setup.cfg memory
	$(PYTHON) -m pylint --rcfile ../pylintrc memory --score=n
	$(PYTHON) -m isort . --check-only --diff
	$(BLACK) --check --diff

.PHONY: build
build:
	$(MAKE) -C ../docker hail-ubuntu
	python3 ../ci/jinja2_render.py '{"hail_ubuntu_image":{"image":"'$$(cat ../docker/hail-ubuntu-image-ref)'"}}' Dockerfile Dockerfile.out
	../docker-build.sh .. memory/Dockerfile.out $(MEMORY_IMAGE)

.PHONY: deploy
deploy: build
	! [ -z $(NAMESPACE) ]  # call this like: make deploy NAMESPACE=default
	kubectl -n $(NAMESPACE) apply -f service-account.yaml
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":$(DEPLOY),"memory_image":{"image":"$(MEMORY_IMAGE)"},"global":{"docker_prefix":"$(DOCKER_PREFIX)"},"default_ns":{"name":"$(NAMESPACE)"}}' deployment.yaml deployment.yaml.out
	kubectl -n $(NAMESPACE) apply -f deployment.yaml.out
