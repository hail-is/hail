include ../config.mk

EXTRA_PYTHONPATH := ../hail/python:../gear
PYTHON := PYTHONPATH=$${PYTHONPATH:+$${PYTHONPATH}:}$(EXTRA_PYTHONPATH) python3

TOKEN = $(shell cat /dev/urandom | LC_ALL=C tr -dc 'a-z0-9' | head -c 12)

QUERY_IMAGE := $(DOCKER_PREFIX)/query:$(TOKEN)

.PHONY: check
check:
	$(PYTHON) -m flake8  --config ../setup.cfg query
	$(PYTHON) -m pylint --rcfile ../pylintrc query --score=n

.PHONY: jar
jar:
	$(MAKE) -C ../hail shadowJar
# janky
	cp ../hail/build/libs/hail-all-spark.jar ./hail.jar

HAIL_TEST_GCS_TOKEN := $(shell whoami)
HAIL_REVISION := $(shell git rev-parse HEAD)

.PHONY: push-jar
push-jar: jar
	echo gs://hail-test-dmk9z/$(HAIL_TEST_GCS_TOKEN)/jars/$(HAIL_REVISION).jar >last_uploaded_jar
	gsutil cp ./hail.jar $$(cat last_uploaded_jar)

upload_resources_dir:
	gsutil -m rsync -r ../hail/src/test/resources/ gs://hail-test-dmk9z/$(HAIL_TEST_GCS_TOKEN)/hail-test-resources/test/resources/ & \
		gsutil -m rsync -r ../hail/python/hail/docs/data/ gs://hail-test-dmk9z/$(HAIL_TEST_GCS_TOKEN)/hail-test-resources/doctest/data/ & \
		wait
	touch upload_resources_dir

.PHONY: test
test: push-jar upload_resources_dir
	HAIL_TEST_RESOURCES_DIR='gs://hail-test-dmk9z/$(HAIL_TEST_GCS_TOKEN)/hail-test-resources/test/resources/' \
    HAIL_DOCTEST_DATA_DIR='gs://hail-test-dmk9z/$(HAIL_TEST_GCS_TOKEN)/hail-test-resources/test/resources/' \
    HAIL_SHA='$(HAIL_REVISION)-$(TOKEN)' \
    HAIL_JAR_URL=$$(cat last_uploaded_jar) \
    $(MAKE) -C ../hail pytest

