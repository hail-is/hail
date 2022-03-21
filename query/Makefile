include ../config.mk

EXTRA_PYTHONPATH := ../hail/python:../gear
PYTHON := PYTHONPATH=$${PYTHONPATH:+$${PYTHONPATH}:}$(EXTRA_PYTHONPATH) python3

TOKEN = $(shell cat /dev/urandom | LC_ALL=C tr -dc 'a-z0-9' | head -c 12)

.PHONY: jar
jar:
	$(MAKE) -C ../hail shadowJar
# janky
	cp ../hail/build/libs/hail-all-spark.jar ./hail.jar

HAIL_TEST_GCS_TOKEN := $(shell whoami)
HAIL_TEST_RESOURCES_PREFIX := gs://hail-test-dmk9z/$(HAIL_TEST_GCS_TOKEN)/hail-test-resources
HAIL_TEST_RESOURCES_DIR := $(HAIL_TEST_RESOURCES_PREFIX)/test/resources/
HAIL_DOCTEST_DATA_DIR := $(HAIL_TEST_RESOURCES_PREFIX)/doctest/data/
HAIL_REVISION := $(shell git rev-parse HEAD)
ifeq ($(NAMESPACE),default)
JAR_LOCATION := gs://hail-query/jars/$(HAIL_TEST_GCS_TOKEN)/$(HAIL_REVISION).jar
else
JAR_LOCATION := gs://hail-test-dmk9z/$(HAIL_TEST_GCS_TOKEN)/jars/$(HAIL_REVISION).jar
endif

.PHONY: push-jar
push-jar: jar
	gsutil cp ./hail.jar "$(JAR_LOCATION)"
	echo >last_uploaded_jar "$(JAR_LOCATION)"

upload-resources-dir:
	python3 -m hailtop.aiotools.copy 'null' '[{"from":"../hail/src/test/resources","to":"$(HAIL_TEST_RESOURCES_DIR)"},{"from":"../hail/python/hail/docs/data","to":"$(HAIL_DOCTEST_DATA_DIR)"}]'
	touch upload-resources-dir

.PHONY: test
test: push-jar upload-resources-dir
	HAIL_QUERY_BACKEND=service \
	HAIL_TEST_RESOURCES_DIR='$(HAIL_TEST_RESOURCES_DIR)' \
	HAIL_DOCTEST_DATA_DIR='$(HAIL_DOCTEST_DATA_DIR)' \
	HAIL_SHA='$(HAIL_REVISION)-$(TOKEN)' \
	HAIL_JAR_URL=$$(cat last_uploaded_jar) \
	$(MAKE) -C ../hail pytest

.PHONY: ipython
ipython: push-jar
	HAIL_QUERY_BACKEND=service \
	HAIL_SHA='$(HAIL_REVISION)-$(TOKEN)' \
	HAIL_JAR_URL=$$(cat last_uploaded_jar) \
	ipython

.PHONY: test-no-deps
test-no-deps:
	HAIL_QUERY_BACKEND=service \
	HAIL_TEST_RESOURCES_DIR='$(HAIL_TEST_RESOURCES_DIR)' \
	HAIL_DOCTEST_DATA_DIR='$(HAIL_DOCTEST_DATA_DIR)' \
	HAIL_SHA='$(HAIL_REVISION)-$(TOKEN)' \
	HAIL_JAR_URL=$$(cat last_uploaded_jar) \
	$(MAKE) -C ../hail pytest

.PHONY: ipython-no-deps
ipython-no-deps:
	HAIL_QUERY_BACKEND=service \
	HAIL_SHA='$(HAIL_REVISION)-$(TOKEN)' \
	HAIL_JAR_URL=$$(cat last_uploaded_jar) \
	ipython
