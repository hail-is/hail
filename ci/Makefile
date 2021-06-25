include ../config.mk

TOKEN = $(shell cat /dev/urandom | LC_ALL=C tr -dc 'a-z0-9' | head -c 12)

CI_UTILS_IMAGE := $(DOCKER_PREFIX)/ci-utils:$(TOKEN)
CI_IMAGE := $(DOCKER_PREFIX)/ci:$(TOKEN)
HAIL_BUILDKIT_IMAGE := $(DOCKER_PREFIX)/hail-buildkit:$(TOKEN)

EXTRA_PYTHONPATH := ../batch:../hail/python:../gear:../web_common
PYTHON := PYTHONPATH=$${PYTHONPATH:+$${PYTHONPATH}:}$(EXTRA_PYTHONPATH) python3

BLACK := $(PYTHON) -m black ci --line-length=120 --skip-string-normalization

.PHONY: check
check:
	$(PYTHON) -m flake8 ci
	$(PYTHON) -m pylint --rcfile ../pylintrc ci --score=n
	$(BLACK) --check --diff
	curlylint .
	bash ../check-sql.sh

.PHONY: blacken
blacken:
	$(BLACK)

.PHONY: service-base
service-base:
	$(MAKE) -C ../docker service-base

.PHONY: build-ci-utils
build-ci-utils: service-base
	python3 jinja2_render.py '{"service_base_image":{"image":"'$$(cat ../docker/service-base-image-ref)'"}}' Dockerfile.ci-utils Dockerfile.ci-utils.out
	../docker-build.sh . Dockerfile.ci-utils.out $(CI_UTILS_IMAGE)

.PHONY: build-hail-buildkit
build-hail-buildkit:
	../docker-build.sh . buildkit/Dockerfile $(HAIL_BUILDKIT_IMAGE)

.PHONY: build
build: service-base
	python3 jinja2_render.py '{"service_base_image":{"image":"'$$(cat ../docker/service-base-image-ref)'"}}' Dockerfile Dockerfile.out
	../docker-build.sh .. ci/Dockerfile.out $(CI_IMAGE)

.PHONY: deploy
deploy: build build-ci-utils build-hail-buildkit
	! [ -z $(NAMESPACE) ]  # call this like: make deploy NAMESPACE=default
	! [ -z "$(CI_DEVELOPER_TEST_REPO_TOKEN)" -a $(NAMESPACE) != "default" ]  # for dev namespaces, you must specify a github repo by its token, check your currently running CI's value for HAIL_WATCHED_BRANCHES
	python3 jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":$(DEPLOY),"default_ns":{"name":"$(NAMESPACE)"},"ci_image":{"image":"$(CI_IMAGE)"},"global":{"project":"$(PROJECT)","zone":"$(ZONE)","ip":"$(IP)","domain":"$(DOMAIN)", "k8s_server_url":"$(KUBERNETES_SERVER_URL)"},"ci_utils_image":{"image":"$(CI_UTILS_IMAGE)"},"ci_database":{"user_secret_name":"sql-ci-user-config"},"hail_buildkit_image":{"image":"$(HAIL_BUILDKIT_IMAGE)"},"create_ci_test_repo":{"token":"$(CI_DEVELOPER_TEST_REPO_TOKEN)"}}' deployment.yaml deployment.yaml.out
	kubectl -n $(NAMESPACE) apply -f deployment.yaml.out
