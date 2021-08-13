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
JAR_LOCATION := gs://hail-test-dmk9z/$(HAIL_TEST_GCS_TOKEN)/jars/$(HAIL_REVISION).jar

.PHONY: push-jar
push-jar: jar
	gsutil cp ./hail.jar "$(JAR_LOCATION)"
	echo >last_uploaded_jar "$(JAR_LOCATION)"

upload-resources-dir:
	gsutil -m rsync -r ../hail/src/test/resources/ gs://hail-test-dmk9z/$(HAIL_TEST_GCS_TOKEN)/hail-test-resources/test/resources/ & \
		gsutil -m rsync -r ../hail/python/hail/docs/data/ gs://hail-test-dmk9z/$(HAIL_TEST_GCS_TOKEN)/hail-test-resources/doctest/data/ & \
		wait
	touch upload-resources-dir

.PHONY: test
test: push-jar upload-resources-dir
	HAIL_TEST_RESOURCES_DIR='gs://hail-test-dmk9z/$(HAIL_TEST_GCS_TOKEN)/hail-test-resources/test/resources/' \
    HAIL_DOCTEST_DATA_DIR='gs://hail-test-dmk9z/$(HAIL_TEST_GCS_TOKEN)/hail-test-resources/doctest/data/' \
    HAIL_SHA='$(HAIL_REVISION)-$(TOKEN)' \
    HAIL_JAR_URL=$$(cat last_uploaded_jar) \
    $(MAKE) -C ../hail pytest

.PHONY: ipython
ipython: push-jar
	HAIL_SHA='$(HAIL_REVISION)-$(TOKEN)' \
	HAIL_JAR_URL=$$(cat last_uploaded_jar) \
	ipython

.PHONY: test-no-deps
test-no-deps:
	HAIL_TEST_RESOURCES_DIR='gs://hail-test-dmk9z/$(HAIL_TEST_GCS_TOKEN)/hail-test-resources/test/resources/' \
    HAIL_DOCTEST_DATA_DIR='gs://hail-test-dmk9z/$(HAIL_TEST_GCS_TOKEN)/hail-test-resources/doctest/data/' \
    HAIL_SHA='$(HAIL_REVISION)-$(TOKEN)' \
    HAIL_JAR_URL=$$(cat last_uploaded_jar) \
    $(MAKE) -C ../hail pytest

.PHONY: ipython-no-deps
ipython-no-deps:
	HAIL_SHA='$(HAIL_REVISION)-$(TOKEN)' \
	HAIL_JAR_URL=$$(cat last_uploaded_jar) \
	ipython

