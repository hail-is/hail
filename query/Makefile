include ../config.mk

QUERY_STORAGE_URI := $(shell kubectl get secret global-config --template={{.data.query_storage_uri}} | base64 --decode)
TEST_STORAGE_URI := $(shell kubectl get secret global-config --template={{.data.test_storage_uri}} | base64 --decode)

EXTRA_PYTHONPATH := ../hail/python:../gear
PYTHON := PYTHONPATH=$${PYTHONPATH:+$${PYTHONPATH}:}$(EXTRA_PYTHONPATH) python3

TOKEN = $(shell cat /dev/urandom | LC_ALL=C tr -dc 'a-z0-9' | head -c 12)

.PHONY: jar
jar:
	$(MAKE) -C ../hail shadowJar
# janky
	cp ../hail/build/libs/hail-all-spark.jar ./hail.jar

HAIL_TEST_GCS_TOKEN := $(shell whoami)
HAIL_TEST_RESOURCES_PREFIX := $(TEST_STORAGE_URI)/$(HAIL_TEST_GCS_TOKEN)/hail-test-resources
HAIL_TEST_RESOURCES_DIR := $(HAIL_TEST_RESOURCES_PREFIX)/test/resources/
HAIL_DOCTEST_DATA_DIR := $(HAIL_TEST_RESOURCES_PREFIX)/doctest/data/
HAIL_REVISION := $(shell git rev-parse HEAD)
ifeq ($(NAMESPACE),default)
ifeq ($(DEPLOY_JAR_FOR_PUBLIC_USE),true)
# This should only be used if the normal CI deploy process fails and you need to upload a JAR to the
# expected location for our users.
JAR_LOCATION := $(QUERY_STORAGE_URI)/jars/$(HAIL_REVISION).jar
else
JAR_LOCATION := $(QUERY_STORAGE_URI)/jars/$(HAIL_TEST_GCS_TOKEN)/$(HAIL_REVISION).jar
endif
else
JAR_LOCATION := $(TEST_STORAGE_URI)/$(NAMESPACE)/jars/$(HAIL_REVISION).jar
endif

.PHONY: upload-query-jar
upload-query-jar: jar
	! [ -z $(NAMESPACE) ]  # call this like: make deploy NAMESPACE=default
	gsutil cp ./hail.jar "$(JAR_LOCATION)"
	echo >last_uploaded_jar "$(JAR_LOCATION)"

upload-resources-dir:
	! [ -z $(NAMESPACE) ]  # call this like: make deploy NAMESPACE=default
	python3 -m hailtop.aiotools.copy 'null' '[{"from":"../hail/src/test/resources","to":"$(HAIL_TEST_RESOURCES_DIR)"},{"from":"../hail/python/hail/docs/data","to":"$(HAIL_DOCTEST_DATA_DIR)"}]'
	touch upload-resources-dir

.PHONY: test
test: upload-query-jar upload-resources-dir
	HAIL_QUERY_BACKEND=batch \
	HAIL_TEST_RESOURCES_DIR='$(HAIL_TEST_RESOURCES_DIR)' \
	HAIL_DOCTEST_DATA_DIR='$(HAIL_DOCTEST_DATA_DIR)' \
	HAIL_JAR_URL=$$(cat last_uploaded_jar) \
	$(MAKE) -C ../hail pytest

.PHONY: ipython
ipython: upload-query-jar
	HAIL_QUERY_BACKEND=batch \
	HAIL_JAR_URL=$$(cat last_uploaded_jar) \
	ipython

.PHONY: test-no-deps
test-no-deps:
	HAIL_QUERY_BACKEND=batch \
	HAIL_TEST_RESOURCES_DIR='$(HAIL_TEST_RESOURCES_DIR)' \
	HAIL_DOCTEST_DATA_DIR='$(HAIL_DOCTEST_DATA_DIR)' \
	HAIL_JAR_URL=$$(cat last_uploaded_jar) \
	$(MAKE) -C ../hail pytest

.PHONY: ipython-no-deps
ipython-no-deps:
	HAIL_QUERY_BACKEND=batch \
	HAIL_JAR_URL=$$(cat last_uploaded_jar) \
	ipython
