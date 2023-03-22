include ../config.mk

CI_UTILS_IMAGE := $(DOCKER_PREFIX)/ci-utils:$(TOKEN)
CI_IMAGE := $(DOCKER_PREFIX)/ci:$(TOKEN)
HAIL_BUILDKIT_IMAGE := $(DOCKER_PREFIX)/hail-buildkit:$(TOKEN)

EXTRA_PYTHONPATH := ../batch:../hail/python:../gear:../web_common
PYTHON := PYTHONPATH=$${PYTHONPATH:+$${PYTHONPATH}:}$(EXTRA_PYTHONPATH) python3

BLACK := $(PYTHON) -m black . --line-length=120 --skip-string-normalization

.PHONY: check
check:
	$(PYTHON) -m flake8 .
	$(PYTHON) -m pylint --rcfile ../pylintrc ci --score=n
	$(PYTHON) -m mypy --config-file ../setup.cfg .
	$(PYTHON) -m isort . --check-only --diff
	$(BLACK) --check --diff
	curlylint .
	bash ../check-sql.sh

.PHONY: blacken
blacken:
	$(BLACK)

.PHONY: hail-ubuntu
hail-ubuntu:
	$(MAKE) -C ../docker hail-ubuntu

.PHONY: build-ci-utils
build-ci-utils: service-base
	python3 jinja2_render.py '{"service_base_image":{"image":"'$$(cat ../docker/service-base-image-ref)'"}}' Dockerfile.ci-utils Dockerfile.ci-utils.out
	../docker-build.sh . Dockerfile.ci-utils.out $(CI_UTILS_IMAGE)

.PHONY: build-hail-buildkit
build-hail-buildkit:
	python3 jinja2_render.py '{"global":{"docker_prefix":"$(DOCKER_PREFIX)"}}' buildkit/Dockerfile buildkit/Dockerfile.out
	../docker-build.sh . buildkit/Dockerfile.out $(HAIL_BUILDKIT_IMAGE)

.PHONY: build
build: hail-ubuntu
	python3 jinja2_render.py '{"hail_ubuntu_image":{"image":"'$$(cat ../docker/hail-ubuntu-image-ref)'"}}' Dockerfile Dockerfile.out
	../docker-build.sh .. ci/Dockerfile.out $(CI_IMAGE)

.PHONY: deploy
deploy: build build-ci-utils build-hail-buildkit
	! [ -z $(NAMESPACE) ]  # call this like: make deploy NAMESPACE=default
	python3 jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":$(DEPLOY),"default_ns":{"name":"$(NAMESPACE)"},"ci_image":{"image":"$(CI_IMAGE)"},"ci_utils_image":{"image":"$(CI_UTILS_IMAGE)"},"ci_database":{"user_secret_name":"sql-ci-user-config"},"hail_buildkit_image":{"image":"$(HAIL_BUILDKIT_IMAGE)"}}' deployment.yaml deployment.yaml.out
	kubectl -n $(NAMESPACE) apply -f deployment.yaml.out
