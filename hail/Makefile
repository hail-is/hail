.PHONY: jars clean

include env_var.mk
.DEFAULT_GOAL := shadowJar
MAKEFLAGS += --no-builtin-rules
.SUFFIXES:

REVISION := $(shell git rev-parse HEAD)
SHORT_REVISION := $(shell git rev-parse --short=12 HEAD)
BRANCH := $(shell git rev-parse --abbrev-ref HEAD)
URL := $(shell git config --get remote.origin.url)
SPARK_VERSION := 2.4.0
HAIL_MAJOR_MINOR_VERSION := 0.2
HAIL_PIP_VERSION := $(HAIL_MAJOR_MINOR_VERSION).21

$(eval $(call ENV_VAR,REVISION))
$(eval $(call ENV_VAR,SHORT_REVISION))
$(eval $(call ENV_VAR,BRANCH))
$(eval $(call ENV_VAR,URL))
$(eval $(call ENV_VAR,SPARK_VERSION))
$(eval $(call ENV_VAR,HAIL_PIP_VERSION))

HAIL_PYTHON3 ?= python3
PIP ?= $(HAIL_PYTHON3) -m pip

# not perfect, not robust to simdpp changes, but probably fine
JAR_SOURCES := $(shell git ls-files src/main)
JAR_SOURCES += build.gradle
JAR_TEST_SOURCES := $(shell git ls-files src/test)
PY_FILES := $(shell git ls-files python)

INIT_SCRIPTS := python/hailtop/hailctl/deploy.yaml
PYTHON_VERSION_INFO := python/hail/hail_version python/hail/hail_pip_version python/hailtop/hailctl/hail_version
SCALA_BUILD_INFO := src/main/resources/build-info.properties

SHADOW_JAR := build/libs/hail-all-spark.jar
SHADOW_TEST_JAR := build/libs/hail-all-spark-test.jar
WHEEL := build/deploy/dist/hail-$(HAIL_PIP_VERSION)-py3-none-any.whl

.PHONY: shadowJar
shadowJar: $(SHADOW_JAR)

GRADLE_ARGS += -Dspark.version=$(SPARK_VERSION) -Dpy4j.version$(PY4J_VERSION)

ifdef HAIL_COMPILE_NATIVES
$(SHADOW_JAR): native-lib-prebuilt
endif
$(SHADOW_JAR): $(SCALA_BUILD_INFO) $(JAR_SOURCES)
	./gradlew shadowJar $(GRADLE_ARGS)

ifdef HAIL_COMPILE_NATIVES
$(SHADOW_TEST_JAR): native-lib-prebuilt
endif
$(SHADOW_TEST_JAR): $(SCALA_BUILD_INFO) $(JAR_SOURCES) $(JAR_TEST_SOURCES)
	./gradlew shadowTestJar $(GRADLE_ARGS)

jars: $(SHADOW_JAR) $(SHADOW_TEST_JAR)

.PHONY: jvm-test
ifdef HAIL_COMPILE_NATIVES
jvm-test: native-lib-prebuilt
endif
jvm-test: $(SCALA_BUILD_INFO) $(JAR_SOURCES) $(JAR_TEST_SOURCES)
	+./pgradle test $(GRADLE_ARGS) $(GRADLE_TEST_ARGS)

src/main/resources/build-info.properties: env/REVISION env/SHORT_REVISION env/BRANCH env/URL
src/main/resources/build-info.properties: env/SPARK_VERSION env/HAIL_PIP_VERSION
src/main/resources/build-info.properties: Makefile
	echo '[Build Metadata]' > $@
	echo 'user=$(USER)' >> $@
	echo 'revision=$(REVISION)' >> $@
	echo 'branch=$(BRANCH)' >> $@
	echo 'date=$(shell date -u +%Y-%m-%dT%H:%M:%SZ)' >> $@
	echo 'url=$(URL)' >> $@
	echo 'sparkVersion=$(SPARK_VERSION)' >> $@
	echo 'hailPipVersion=$(HAIL_PIP_VERSION)' >> $@

.PHONY: python-version-info
python-version-info: $(PYTHON_VERSION_INFO)

python/hail/hail_version: env/SHORT_REVISION env/HAIL_PIP_VERSION
	echo $(HAIL_PIP_VERSION)-$(SHORT_REVISION) > $@

python/hail/hail_pip_version: env/HAIL_PIP_VERSION
	echo $(HAIL_PIP_VERSION) > $@

python/hailtop/hailctl/hail_version: python/hail/hail_version
	cp -f $< $@

python/README.md: ../README.md
	cp ../README.md python/

python/hail/hail-all-spark.jar: $(SHADOW_JAR)
	cp -f $< $@

.PHONY: pytest
pytest: $(PYTHON_VERSION_INFO) $(SHADOW_JAR) $(INIT_SCRIPTS)
pytest: python/README.md python/hail/hail-all-spark.jar
	cd python && $(HAIL_PYTHON3) setup.py pytest \
     --addopts "-v \
     --color=no \
     --instafail \
     -r A \
     --self-contained-html \
     --html=../build/reports/pytest.html \
     --noconftest \
     $(PYTEST_ARGS)"

.PHONY: doctest
doctest: $(PYTHON_VERSION_INFO) $(SHADOW_JAR) $(INIT_SCRIPTS)
doctest: python/README.md python/hail/hail-all-spark.jar
	cd python && $(HAIL_PYTHON3) setup.py pytest \
	  --addopts "-v \
    --color=no \
    --instafail \
    -r A \
    --self-contained-html \
    --html=../build/reports/pytest-doctest.html \
    --doctest-modules \
    --doctest-glob='*.rst' \
    --ignore=setup.py \
    --ignore=setup-hailtop.py \
    --ignore=hail/docs/conf.py \
    --ignore=cluster-tests \
    --ignore=hailtop \
    --ignore=test \
    --ignore=hail/docs/doctest_write_data.py \
    --ignore=hail/docs/getting_started_developing.rst \
    --ignore=hail/docs/getting_started.rst \
    $(PYTEST_ARGS)"

.PHONY: wheel
wheel: $(WHEEL)

$(WHEEL): $(PYTHON_VERSION_INFO) $(SHADOW_JAR) $(INIT_SCRIPTS) $(PY_FILES)
	rm -rf build/deploy
	mkdir -p build/deploy
	mkdir -p build/deploy/src
	cp ../README.md build/deploy/
	rsync -rv \
	    --exclude '__pycache__/' \
	    --exclude 'docs/' \
	    --exclude 'test/' \
	    --exclude '*.log' \
	    python/ build/deploy/
	cp build/libs/hail-all-spark.jar build/deploy/hail/
	cd build/deploy; $(HAIL_PYTHON3) setup.py sdist bdist_wheel

# if the DEPLOY_PROD flag is not set, then deploy init scripts into a dev-username location
ifndef DEPLOY_PROD
DEV_CLARIFIER := $(shell whoami)-dev/
CLOUD_SUB_FOLDER := $(HAIL_PIP_VERSION)-$(SHORT_REVISION)
UPLOAD_RETENTION =
else
CLOUD_SUB_FOLDER := $(HAIL_PIP_VERSION)
UPLOAD_RETENTION = gsutil -m retention temp set "$(cloud_base)/*"
endif

HAILCTL_BUCKET_BASE ?= gs://hail-common/hailctl/dataproc

cloud_base := $(HAILCTL_BUCKET_BASE)/$(DEV_CLARIFIER)$(CLOUD_SUB_FOLDER)
wheel_cloud_path := $(cloud_base)/hail-$(HAIL_PIP_VERSION)-py3-none-any.whl
resources := $(wildcard python/hailtop/hailctl/dataproc/resources/*)
$(eval $(call ENV_VAR,cloud_base))
$(eval $(call ENV_VAR,wheel_cloud_path))

python/hailtop/hailctl/deploy.yaml: env/cloud_base env/wheel_cloud_path
python/hailtop/hailctl/deploy.yaml: $(resources) python/requirements.txt
	rm -f $@
	echo "dataproc:" >> $@
	for FILE in $(notdir $(resources)); do \
	  echo "  $$FILE: $(cloud_base)/$$FILE" >> $@ || exit 1; done
	echo "  wheel: $(wheel_cloud_path)" >> $@
	echo "  pip_dependencies: $(shell cat python/requirements.txt | grep -v pyspark | tr "\n" "|||")" >> $@

.PHONY: upload-artifacts
upload-artifacts: $(WHEEL)
	gsutil -m cp -r $(resources) $(WHEEL) $(cloud_base)
	gsutil -m acl set -r public-read $(cloud_base)
	$(UPLOAD_RETENTION)

.PHONY: install-editable
install-editable: $(PYTHON_VERSION_INFO) $(INIT_SCRIPTS)
install-editable: python/README.md python/hail/hail-all-spark.jar
	-$(PIP) uninstall -y hail
	cd python && $(PIP) install -e .

.PHONY: install
install: $(WHEEL)
	-$(PIP) uninstall -y hail
	$(PIP) install $(WHEEL)

.PHONY: install-on-cluster
install-on-cluster: $(WHEEL)
	sed '/^pyspark/d' python/requirements.txt | xargs $(PIP) install -U
	-$(PIP) uninstall -y hail
	$(PIP) install $(WHEEL) --no-deps

.PHONY: install-hailctl
install-hailctl: install upload-artifacts

cluster_name := cluster-$(shell whoami)-$(shell echo $$RANDOM)
cluster_test_files := $(wildcard python/cluster-tests/*.py)
.PHONY: test-dataproc
test-dataproc: install-hailctl
	hailctl dataproc start $(cluster_name) --max-idle 10m --vep GRCh37
	for FILE in $(cluster_test_files); do \
	  hailctl dataproc submit $(cluster_name) $$FILE || exit 1; done || exit


DEPLOYED_VERSION = $(shell \
  $(PIP) --no-cache-dir search hail \
   | grep '^hail ' \
   | sed 's/hail (//' \
   | sed 's/).*//')
.PHONY: check-pypi
check-pypi:
	if [ -z "$$DEPLOY_PROD" ]; then \
	  echo "DEPLOY_PROD must be set to deploy to PyPI"; exit 1; fi
	if [ "$(DEPLOYED_VERSION)" == "$(HAIL_PIP_VERSION)" ]; then \
	  echo "version $(HAIL_PIP_VERSION) already deployed"; exit 1; fi

HAIL_TWINE_CREDS_FOLDER ?= /secrets/

.PHONY: pypi-deploy
pypi-deploy: check-pypi test-dataproc set-docs-sha
	TWINE_USERNAME=$(shell cat $(HAIL_TWINE_CREDS_FOLDER)/pypi-username) \
	TWINE_PASSWORD=$(shell cat $(HAIL_TWINE_CREDS_FOLDER)/pypi-password) \
	twine upload build/deploy/dist/*

TAG_EXISTS = $(shell git ls-remote --exit-code --tags origin $(HAIL_PIP_VERSION) || echo "does not exist")
.PHONY: check-tag
check-tag:
	if [ -z "$(TAG_EXISTS)" ]; then echo "tag $(HAIL_PIP_VERSION) already exists"; exit 1; fi

.PHONY: tag
tag: check-tag pypi-deploy
	git tag $(HAIL_PIP_VERSION) -m "Hail version $(HAIL_PIP_VERSION)"
	git push https://github.com/hail-is/hail.git $(HAIL_PIP_VERSION)

docs_location := gs://hail-common/builds/0.2/docs/hail-0.2-docs-$(REVISION).tar.gz
local_sha_location := build/deploy/latest-hash-spark-$(SPARK_VERSION).txt
cloud_sha_location := gs://hail-common/builds/0.2/latest-hash/cloudtools-5-spark-2.4.0.txt
.PHONY: set-docs-sha
set-docs-sha:
	mkdir -p $(dir $(local_sha_location))
	gsutil ls $(docs_location)  # make sure file exists
	echo "$(REVISION)" > $(local_sha_location)
	gsutil cp $(local_sha_location) $(cloud_sha_location)
	gsutil acl set public-read $(cloud_sha_location)

.PHONY: deploy
deploy: tag

.PHONY: install-deps
install-deps:
	$(PIP) install -U -r python/requirements.txt -r python/dev-requirements.txt

BENCHMARK_DOCKER_TAG := benchmark_$(shell whoami)
PROJECT = $(shell gcloud config get-value project)
BENCHMARK_LATEST = gcr.io/$(PROJECT)/$(BENCHMARK_DOCKER_TAG):latest

.PHONY: build-benchmark-image
build-benchmark-image: wheel
	cp $(WHEEL) benchmark
	cp python/requirements.txt benchmark
	cd benchmark && docker build -t $(BENCHMARK_DOCKER_TAG) . --build-arg WHEEL=$(notdir $(WHEEL))
	rm benchmark/$(notdir $(WHEEL))
	rm benchmark/requirements.txt

.PHONY: push-benchmark-image
push-benchmark-image: build-benchmark-image
	docker tag $(BENCHMARK_DOCKER_TAG) $(BENCHMARK_LATEST)
	docker push $(BENCHMARK_LATEST)

BENCHMARK_ITERS ?= 5
BENCHMARK_REPLICATES ?= 5
.PHONY: benchmark
benchmark: push-benchmark-image install
	$(HAIL_PYTHON3) benchmark/benchmark_in_pipeline.py $(BENCHMARK_LATEST) gs://hail-benchmarks/$(shell whoami) $(HAIL_PIP_VERSION)-$(SHORT_REVISION) $(BENCHMARK_REPLICATES) $(BENCHMARK_ITERS)

clean: clean-env clean-docs native-lib-clean
	./gradlew clean $(GRADLE_ARGS)
	rm -rf build/
	rm -rf python/hail/hail-all-spark.jar
	rm -rf python/README.md

.PHONY: clean-docs
clean-docs:
	$(MAKE) -C www clean

HAIL_CACHE_VERSION = $(shell cat python/hail/hail_version)

.PHONY: docs
docs: install $(PYTHON_VERSION_INFO)
	mkdir -p build
	rm -rf build/www
	cp -R www build/www
	mkdir -p build/docs
	cp -R python/hail/docs build/
	cp python/hail/hail_version build/
	sed -E "s/\(hail\#([0-9]+)\)/(\[#\1](https:\/\/github.com\/hail-is\/hail\/pull\/\1))/g" \
    < build/docs/change_log.md \
    | pandoc -o build/docs/change_log.rst
	$(MAKE) -C build/www
	$(MAKE) SPHINXOPTS='-tchecktutorial' -C build/docs BUILDDIR=_build clean html
	mkdir -p build/www/docs
	rm -rf build/www/docs/$(HAIL_MAJOR_MINOR_VERSION)
	mv build/docs/_build/html build/www/docs/$(HAIL_MAJOR_MINOR_VERSION)
	find ./build/www -iname *.html -type f -exec sed -i '' -e "s/\.css/\.css\?v\=$(HAIL_CACHE_VERSION)/" {} +;
	@echo Built docs: build/www/docs/$(HAIL_MAJOR_MINOR_VERSION)/index.html

.PHONY: upload-docs
upload-docs: docs
	cd build && tar czf www.tar.gz www
	gsutil -m cp build/www.tar.gz $(docs_location)
	gsutil -m retention temp set $(docs_location)
	gsutil -m acl set public-read $(docs_location)

.PHONY: test
test: pytest jvm-test doctest tutorial-test

.PHONY: tutorial-test
docs-no-test: install $(PYTHON_VERSION_INFO)
	mkdir -p build
	rm -rf build/www
	cp -R www build/www
	mkdir -p build/docs
	cp -R python/hail/docs build/
	cp python/hail/hail_version build/
	sed -E "s/\(hail\#([0-9]+)\)/(\[#\1](https:\/\/github.com\/hail-is\/hail\/pull\/\1))/g" \
    < build/docs/change_log.md \
    | pandoc -o build/docs/change_log.rst
	$(MAKE) -C build/www
	$(MAKE) -C build/docs BUILDDIR=_build clean html
	mkdir -p build/www/docs
	rm -rf build/www/docs/$(HAIL_MAJOR_MINOR_VERSION)
	mv build/docs/_build/html build/www/docs/$(HAIL_MAJOR_MINOR_VERSION)
	find ./build/www -iname *.html -type f -exec sed -i '' -e "s/\.css/\.css\?v\=$(HAIL_CACHE_VERSION)/" {} +;
	@echo Built docs: build/www/docs/$(HAIL_MAJOR_MINOR_VERSION)/index.html

.PHONY: native-lib native-lib-test native-lib-clean native-lib-prebuilt native-lib-reset-prebuilt
native-lib:
	$(MAKE) -C src/main/c
native-lib-test:
	$(MAKE) -C src/main/c test
native-lib-clean:
	$(MAKE) -C src/main/c clean
native-lib-prebuilt:
	$(MAKE) -C src/main/c prebuilt
native-lib-reset-prebuilt:
	$(MAKE) -C src/main/c reset-prebuilt
