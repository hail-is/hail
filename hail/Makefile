.PHONY: jars clean

HAIL_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include env_var.mk

.DEFAULT_GOAL := shadowJar
MAKEFLAGS += --no-builtin-rules
.SUFFIXES:

REVISION := $(shell git rev-parse HEAD)
SHORT_REVISION := $(shell git rev-parse --short=12 HEAD)
BRANCH := $(shell git rev-parse --abbrev-ref HEAD)
SCALA_VERSION ?= 2.12.13
SPARK_VERSION ?= 3.1.2
HAIL_MAJOR_MINOR_VERSION := 0.2
HAIL_PATCH_VERSION := 80
HAIL_PIP_VERSION := $(HAIL_MAJOR_MINOR_VERSION).$(HAIL_PATCH_VERSION)
HAIL_VERSION := $(HAIL_PIP_VERSION)-$(SHORT_REVISION)
ELASTIC_MAJOR_VERSION ?= 7


$(eval $(call ENV_VAR,REVISION))
$(eval $(call ENV_VAR,SHORT_REVISION))
$(eval $(call ENV_VAR,BRANCH))
$(eval $(call ENV_VAR,SPARK_VERSION))
$(eval $(call ENV_VAR,HAIL_PIP_VERSION))
$(eval $(call ENV_VAR,HAIL_DEBUG_MODE))
$(eval $(call ENV_VAR,ELASTIC_MAJOR_VERSION))

HAIL_PYTHON3 ?= python3
PIP ?= $(HAIL_PYTHON3) -m pip
ifdef JAVA_HOME
JAVAC ?= $(JAVA_HOME)/bin/javac
JAR ?= $(JAVA_HOME)/bin/jar
else
JAVAC ?= javac
JAR ?= jar
endif


# not perfect, not robust to simdpp changes, but probably fine
BUILD_DEBUG_PREFIX := build/classes/scala/debug
JAR_SOURCES := $(shell git ls-files src/main)
JAR_SOURCES += build.gradle
JAR_TEST_SOURCES := $(shell git ls-files src/test)
JAR_DEBUG_SOURCES := src/debug/scala/is/hail/annotations/Memory.java
JAR_DEBUG_CLASSES := $(addprefix $(BUILD_DEBUG_PREFIX)/, $(JAR_DEBUG_SOURCES:src/debug/scala/%.java=%.class))
PY_FILES := $(shell git ls-files python)

INIT_SCRIPTS := python/hailtop/hailctl/deploy.yaml
PYTHON_VERSION_INFO := python/hail/hail_version
PYTHON_VERSION_INFO += python/hail/hail_pip_version
PYTHON_VERSION_INFO += python/hailtop/hail_version
PYTHON_VERSION_INFO += python/hail/docs/_static/hail_version.js
SCALA_BUILD_INFO := src/main/resources/build-info.properties

SHADOW_JAR := build/libs/hail-all-spark.jar
SHADOW_TEST_JAR := build/libs/hail-all-spark-test.jar
PYTHON_JAR := python/hail/backend/hail-all-spark.jar
WHEEL := build/deploy/dist/hail-$(HAIL_PIP_VERSION)-py3-none-any.whl
EGG := build/deploy/dist/hail-$(HAIL_PIP_VERSION)-py3.6.egg
ELASTICSEARCH_311_JAR := libs/elasticsearch-spark-30_2.12-8.0.0-SNAPSHOT-custom-hail-spark311.jar

GRADLE_ARGS += -Dscala.version=$(SCALA_VERSION) -Dspark.version=$(SPARK_VERSION) -Delasticsearch.major-version=$(ELASTIC_MAJOR_VERSION)

.PHONY: elasticsearchJar
elasticsearchJar: $(ELASTICSEARCH_311_JAR)

$(ELASTICSEARCH_311_JAR):
	@mkdir -p libs
	curl https://storage.googleapis.com/hail-common/elasticsearch-libs/elasticsearch-spark-30_2.12-8.0.0-SNAPSHOT-custom-hail-spark311.jar >  libs/elasticsearch-spark-30_2.12-8.0.0-SNAPSHOT-custom-hail-spark311.jar

.PHONY: shadowJar
shadowJar: $(SHADOW_JAR)

$(JAR_SOURCES): $(ELASTICSEARCH_311_JAR)
$(JAR_TEST_SOURCES): $(ELASTICSEARCH_311_JAR)

ifdef HAIL_COMPILE_NATIVES
$(SHADOW_JAR): native-lib-prebuilt
endif
ifdef HAIL_DEBUG_MODE
$(SHADOW_JAR): $(JAR_DEBUG_CLASSES)
endif
$(SHADOW_JAR): $(SCALA_BUILD_INFO) $(JAR_SOURCES) env/HAIL_DEBUG_MODE env/ELASTIC_MAJOR_VERSION
	./gradlew shadowJar $(GRADLE_ARGS)
ifdef HAIL_DEBUG_MODE
	$(JAR) -uf $(SHADOW_JAR) -C $(BUILD_DEBUG_PREFIX) is/hail/annotations/Memory.class
endif

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

.PHONY: services-jvm-test
ifdef HAIL_COMPILE_NATIVES
services-jvm-test: native-lib-prebuilt
endif
services-jvm-test: $(SCALA_BUILD_INFO) $(JAR_SOURCES) $(JAR_TEST_SOURCES)
	+./pgradle testServices $(GRADLE_ARGS) $(GRADLE_TEST_ARGS)

# javac args from compileJava in build.gradle
$(BUILD_DEBUG_PREFIX)/%.class: src/debug/scala/%.java
	@mkdir -p $(BUILD_DEBUG_PREFIX)
	$(JAVAC) -d $(BUILD_DEBUG_PREFIX) -Xlint:all -Werror -XDenableSunApiLintControl -Xlint:-sunapi $<

src/main/resources/build-info.properties: env/REVISION env/SHORT_REVISION env/BRANCH
src/main/resources/build-info.properties: env/SPARK_VERSION env/HAIL_PIP_VERSION
src/main/resources/build-info.properties: Makefile
	echo '[Build Metadata]' > $@
	echo 'user=$(USER)' >> $@
	echo 'revision=$(REVISION)' >> $@
	echo 'branch=$(BRANCH)' >> $@
	echo 'date=$(shell date -u +%Y-%m-%dT%H:%M:%SZ)' >> $@
	echo 'sparkVersion=$(SPARK_VERSION)' >> $@
	echo 'hailPipVersion=$(HAIL_PIP_VERSION)' >> $@

.PHONY: python-version-info
python-version-info: $(PYTHON_VERSION_INFO)

python/hail/hail_version: env/SHORT_REVISION env/HAIL_PIP_VERSION
	echo $(HAIL_VERSION) > $@

python/hail/hail_pip_version: env/HAIL_PIP_VERSION
	echo $(HAIL_PIP_VERSION) > $@

python/hail/docs/_static/hail_version.js: python/hail/hail_version python/hail/hail_pip_version
	printf 'hail_version="$(shell cat python/hail/hail_version)";' > $@
	printf 'hail_pip_version="$(shell cat python/hail/hail_pip_version)"' >> $@

python/hailtop/hail_version: python/hail/hail_version
	cp -f $< $@

python/README.md: ../README.md
	cp ../README.md python/

$(PYTHON_JAR): shadowJar
	cp -f $(SHADOW_JAR) $@

.PHONY: pytest
pytest: PYTEST_TARGET ?= test/hail
pytest: $(PYTHON_VERSION_INFO) $(SHADOW_JAR) $(INIT_SCRIPTS)
pytest: python/README.md $(PYTHON_JAR)
	cd python && $(HAIL_PYTHON3) setup.py pytest \
     --addopts "-v \
     --color=no \
     --instafail \
     -r A \
     --self-contained-html \
     --html=../build/reports/pytest.html \
     $(PYTEST_TARGET) \
     $(PYTEST_ARGS)"

.PHONY: doctest
doctest: $(PYTHON_VERSION_INFO) $(SHADOW_JAR) $(INIT_SCRIPTS)
doctest: python/README.md $(PYTHON_JAR)
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

.PHONY: copy-py-files
copy-py-files: $(PYTHON_VERSION_INFO) $(SHADOW_JAR) $(INIT_SCRIPTS) $(PY_FILES) $(PYTHON_JAR)
	rm -rf build/deploy
	mkdir -p build/deploy
	mkdir -p build/deploy/src
	cp ../README.md build/deploy/
	rsync -r \
	    --exclude '.eggs/' \
	    --exclude '.pytest_cache/' \
	    --exclude '__pycache__/' \
	    --exclude 'benchmark_hail/' \
	    --exclude '.mypy_cache/' \
	    --exclude 'docs/' \
	    --exclude 'dist/' \
	    --exclude 'test/' \
	    --exclude '*.log' \
	    python/ build/deploy/

.PHONY: wheel
wheel: $(WHEEL)

$(WHEEL): copy-py-files
	cd build/deploy; $(HAIL_PYTHON3) setup.py -q sdist bdist_wheel

.PHONY: egg
egg: $(EGG)

$(EGG): copy-py-files
	cd build/deploy; $(HAIL_PYTHON3) setup.py -q sdist bdist_egg

# if the DEPLOY_REMOTE flag is not set, then deploy init scripts into a dev-username location
ifndef DEPLOY_REMOTE
DEV_CLARIFIER ?= $(shell whoami)-dev
cloud_base ?= gs://hail-30-day/hailctl/dataproc/$(DEV_CLARIFIER)/$(HAIL_VERSION)
UPLOAD_RETENTION =
else
cloud_base := gs://hail-common/hailctl/dataproc/$(HAIL_PIP_VERSION)
UPLOAD_RETENTION = gsutil -m retention temp set "$(cloud_base)/*"
endif

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
	echo "  pip_dependencies: $(shell cat python/requirements.txt | sed '/^[[:blank:]]*#/d;s/#.*//' | grep -v pyspark | tr "\n" "|||")" >> $@

.PHONY: upload-artifacts
upload-artifacts: $(WHEEL)
	gsutil -m cp -r $(resources) $(WHEEL) $(cloud_base)
	gsutil -m acl set -r public-read $(cloud_base)
	$(UPLOAD_RETENTION)

.PHONY: install-editable
install-editable: $(PYTHON_VERSION_INFO) $(INIT_SCRIPTS)
install-editable: python/README.md $(PYTHON_JAR)
	-$(PIP) uninstall -y hail
	cd python && $(PIP) install -e .

.PHONY: install
install: $(WHEEL)
	-$(PIP) uninstall -y hail
	$(PIP) install $(WHEEL)

.PHONY: install-on-cluster
install-on-cluster: $(WHEEL)
	sed '/^pyspark/d' python/requirements.txt | grep -v '^#' | xargs $(PIP) install -U
	-$(PIP) uninstall -y hail
	$(PIP) install $(WHEEL) --no-deps

.PHONY: install-hailctl
install-hailctl: install upload-artifacts

.PHONY: test-dataproc
test-dataproc: install-hailctl
	bash scripts/test-dataproc.sh

# use curl version >=7.55.0
# set DEPLOY_REMOTE to the hail-is/hail remote
# create an access token with full repo privileges at https://github.com/settings/tokens
# create a file that contains "Authorization: token YOUR_ACCESS_TOKEN_HERE"
# set GITHUB_OAUTH_HEADER_FILE to that filename
.PHONY: deploy
deploy: test-dataproc $(WHEEL)
	bash scripts/deploy.sh $(HAIL_PIP_VERSION) $(HAIL_VERSION) $(REVISION) $(DEPLOY_REMOTE) $(WHEEL) $(GITHUB_OAUTH_HEADER_FILE)

.PHONY: install-dev-deps
install-dev-deps:
	$(PIP) install -U -r python/dev-requirements.txt

.PHONY: install-deps
install-deps: install-dev-deps
	sed "s/^pyspark.*/pyspark==$(SPARK_VERSION)/" python/requirements.txt | xargs $(PIP) install -U

.PHONY: benchmark
benchmark: $(WHEEL)
	HAIL_WHEEL=../hail/$(WHEEL) HAIL_PIP_VERSION=$(HAIL_PIP_VERSION) $(MAKE) -C ../benchmark benchmark

.PHONY: install-benchmark
install-benchmark:
	HAIL_WHEEL=DUMMY HAIL_PIP_VERSION=$(HAIL_PIP_VERSION) $(MAKE) -C ../benchmark install

python/hail/docs/change_log.rst: python/hail/docs/change_log.md
	sed -E "s/\(hail\#([0-9]+)\)/(\[#\1](https:\/\/github.com\/hail-is\/hail\/pull\/\1))/g" \
	  < $< \
	  | pandoc -o $@

ifeq ($(shell uname -s),Darwin)
SED_INPLACE := sed -i ''
else
SED_INPLACE := sed -i
endif

.PHONY: batch-docs
batch-docs:
	$(MAKE) -C python/hailtop/batch/docs \
	      BUILDDIR=$(HAIL_DIR)/build/docs/batch \
	      html
	rm -rf build/www/docs/batch
	mkdir -p build/www/docs
	cp -R build/docs/batch/html build/www/docs/batch
	find build/www/docs/batch \
	     -iname *.html \
	     -type f \
	     -exec $(SED_INPLACE) -e "s/\.css/\.css\?v\=$(HAIL_CACHE_VERSION)/" {} +;
	@echo Built Batch docs: build/www/docs/batch/index.html

HAIL_CACHE_VERSION = $(shell cat python/hail/hail_version)
.PHONY: hail-docs
hail-docs: $(PYTHON_VERSION_INFO) python/hail/docs/change_log.rst
	$(MAKE) -C python/hail/docs \
	        SPHINXOPTS='-tgenerate_notebook_outputs' \
	        BUILDDIR=$(HAIL_DIR)/build/docs/hail \
	        html
	mkdir -p build/www/docs
	rm -rf build/www/docs/$(HAIL_MAJOR_MINOR_VERSION)
	cp -R build/docs/hail/html build/www/docs/$(HAIL_MAJOR_MINOR_VERSION)
	find build/www/docs/$(HAIL_MAJOR_MINOR_VERSION) \
	     -iname *.html \
	     -type f \
	     -exec $(SED_INPLACE) -e "s/\.css/\.css\?v\=$(HAIL_CACHE_VERSION)/" {} +;
	@echo Built Hail docs: build/www/docs/$(HAIL_MAJOR_MINOR_VERSION)/index.html

.PHONY: hail-docs-no-test
hail-docs-no-test: $(PYTHON_VERSION_INFO) python/hail/docs/change_log.rst
	$(MAKE) -C python/hail/docs \
	        BUILDDIR=$(HAIL_DIR)/build/docs/hail \
	        html
	mkdir -p build/www/docs
	rm -rf build/www/docs/$(HAIL_MAJOR_MINOR_VERSION)
	cp -R build/docs/hail/html build/www/docs/$(HAIL_MAJOR_MINOR_VERSION)
	find build/www/docs/$(HAIL_MAJOR_MINOR_VERSION) \
	     -iname *.html \
	     -type f \
	     -exec $(SED_INPLACE) -e "s/\.css/\.css\?v\=$(HAIL_CACHE_VERSION)/" {} +;
	@echo Built docs: file://$(HAIL_DIR)/build/www/docs/$(HAIL_MAJOR_MINOR_VERSION)/index.html

.PHONY: upload-docs
upload-docs: hail-docs batch-docs
	cd build && tar czf www.tar.gz www
	gsutil -m cp build/www.tar.gz $(docs_location)
	gsutil -m retention temp set $(docs_location)
	gsutil -m acl set public-read $(docs_location)

.PHONY: test
test: pytest jvm-test doctest

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

clean-libs:
	rm -rf libs

clean: clean-env clean-libs native-lib-clean
	$(MAKE) -C python/hail/docs clean
	$(MAKE) -C python/hailtop/batch/docs clean
	./gradlew clean $(GRADLE_ARGS)
	rm -rf build/
	rm -rf $(PYTHON_JAR)
	rm -rf python/README.md
	rm -rf $(SCALA_BUILD_INFO)
	rm -rf $(PYTHON_VERSION_INFO)
	rm -rf python/hail/docs/change_log.rst

.PHONY: update-hail-repl
update-hail-repl: NAMESPACE ?= default
update-hail-repl: wheel
	kubectl -n $(NAMESPACE) cp $(WHEEL) $$(kubectl get pods -n $(NAMESPACE) -l app=hail-repl | tail -n +2 | awk '{print $$1}'):.
	kubectl -n $(NAMESPACE) exec -it $$(kubectl get pods -n $(NAMESPACE) -l app=hail-repl | tail -n +2 | awk '{print $$1}') -- pip3 install -U hail-$(HAIL_PIP_VERSION)-py3-none-any.whl
