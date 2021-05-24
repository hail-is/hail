include ../config.mk

EXTRA_PYTHONPATH := ../hail/python:../gear
PYTHON := PYTHONPATH=$${PYTHONPATH:+$${PYTHONPATH}:}$(EXTRA_PYTHONPATH) python3

TOKEN = $(shell cat /dev/urandom | LC_ALL=C tr -dc 'a-z0-9' | head -c 12)

QUERY_IMAGE := $(DOCKER_PREFIX)/query:$(TOKEN)

.PHONY: check
check:
	$(PYTHON) -m flake8  --config ../setup.cfg query
	$(PYTHON) -m pylint --rcfile ../pylintrc query --score=n

.PHONY: build
build:
	$(MAKE) -C ../docker service-java-run-base
	$(MAKE) -C ../hail shadowJar
# janky
	cp ../hail/build/libs/hail-all-spark.jar ./hail.jar
	python3 ../ci/jinja2_render.py '{"service_java_run_base_image":{"image":"'$$(cat ../docker/service-java-run-base-image-ref)'"}}' Dockerfile Dockerfile.out
	../docker-build.sh . Dockerfile.out $(QUERY_IMAGE)

UPLOAD_QUERY_JAR_TOKEN := $(shell cat /dev/urandom | LC_ALL=C tr -dc 'a-z0-9' | head -c 12)
HAIL_REVISION := $(shell git rev-parse HEAD)

.PHONY: deploy
deploy: build
	! [ -z $(NAMESPACE) ]  # call this like: make deploy NAMESPACE=default
	kubectl -n $(NAMESPACE) apply -f service-account.yaml
	gsutil cp ./hail.jar gs://hail-test-dmk9z/$(UPLOAD_QUERY_JAR_TOKEN)/jars/$(HAIL_REVISION).jar
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":$(DEPLOY),"default_ns":{"name":"$(NAMESPACE)"},"query_image":{"image":"$(QUERY_IMAGE)"},"global":{"project":"$(PROJECT)","domain":"$(DOMAIN)"},"upload_query_jar":{"token":"$(UPLOAD_QUERY_JAR_TOKEN)"}}' deployment.yaml deployment.yaml.out
	kubectl -n $(NAMESPACE) apply -f deployment.yaml.out
