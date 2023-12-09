.PHONY: jars clean

HAIL_HAIL_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include env_var.mk
include version.mk

.DEFAULT_GOAL := shadowJar
MAKEFLAGS += --no-builtin-rules
.SUFFIXES:

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

PYTEST_TARGET ?= test/hail

# not perfect, not robust to simdpp changes, but probably fine
BUILD_DEBUG_PREFIX := build/classes/scala/debug
JAR_SOURCES := $(shell git ls-files src/main)
JAR_SOURCES += build.gradle
JAR_TEST_SOURCES := $(shell git ls-files src/test)
JAR_DEBUG_SOURCES := src/debug/scala/is/hail/annotations/Memory.java
JAR_DEBUG_CLASSES := $(addprefix $(BUILD_DEBUG_PREFIX)/, $(JAR_DEBUG_SOURCES:src/debug/scala/%.java=%.class))
PY_FILES := $(shell git ls-files python)

INIT_SCRIPTS := python/hailtop/hailctl/deploy.yaml
PYTHON_VERSION_INFO := python/hail/hail_revision
PYTHON_VERSION_INFO += python/hail/hail_version
PYTHON_VERSION_INFO += python/hail/hail_pip_version
PYTHON_VERSION_INFO += python/hailtop/hail_version
PYTHON_VERSION_INFO += python/hail/docs/_static/hail_version.js

GRADLE_SHADOW_JAR := build/libs/hail-all-spark.jar
GRADLE_NON_SHADOW_JAR := build/libs/hail.jar
GRADLE_SHADOW_TEST_JAR := build/libs/hail-all-spark-test.jar

SHADOW_JAR := build/jars/hail-all-spark.jar
NON_SHADOW_JAR := build/jars/hail.jar
SHADOW_TEST_JAR := build/jars/hail-all-spark-test.jar

PYTHON_JAR := python/hail/backend/hail-all-spark.jar
FAST_PYTHON_JAR := python/hail/backend/hail.jar
FAST_PYTHON_JAR_EXTRA_CLASSPATH := python/hail/backend/extra_classpath
WHEEL := build/deploy/dist/hail-$(HAIL_PIP_VERSION)-py3-none-any.whl

GRADLE_ARGS += -Dscala.version=$(SCALA_VERSION) -Dspark.version=$(SPARK_VERSION) -Delasticsearch.major-version=$(ELASTIC_MAJOR_VERSION)
ifdef GRADLE_CACHE_URI
GRADLE_ARGS += --build-cache -Dorg.gradle.caching.debug=true
endif

TEST_STORAGE_URI = $(shell kubectl get secret global-config --template={{.data.test_storage_uri}} | base64 --decode)
GCP_PROJECT = $(shell kubectl get secret global-config --template={{.data.gcp_project}} | base64 --decode)
CLOUD_HAIL_TEST_RESOURCES_PREFIX = $(TEST_STORAGE_URI)/$(shell whoami)/hail-test-resources
CLOUD_HAIL_TEST_RESOURCES_DIR = $(CLOUD_HAIL_TEST_RESOURCES_PREFIX)/test/resources/
CLOUD_HAIL_DOCTEST_DATA_DIR = $(CLOUD_HAIL_TEST_RESOURCES_PREFIX)/doctest/data/

.PHONY: shadowJar
shadowJar: $(SHADOW_JAR)

$(SHADOW_JAR): $(GRADLE_SHADOW_JAR) build-info.properties
	mkdir -p $(dir $(SHADOW_JAR))
	cp $(GRADLE_SHADOW_JAR) $(SHADOW_JAR)
	$(JAR) -uf $(SHADOW_JAR) build-info.properties

ifdef HAIL_COMPILE_NATIVES
$(GRADLE_SHADOW_JAR): native-lib-prebuilt
endif
ifdef HAIL_DEBUG_MODE
$(GRADLE_SHADOW_JAR): $(JAR_DEBUG_CLASSES)
endif
$(GRADLE_SHADOW_JAR): $(JAR_SOURCES) env/HAIL_DEBUG_MODE env/ELASTIC_MAJOR_VERSION
	./gradlew shadowJar $(GRADLE_ARGS)
ifdef HAIL_DEBUG_MODE
	$(JAR) -uf $(GRADLE_SHADOW_JAR) -C $(BUILD_DEBUG_PREFIX) is/hail/annotations/Memory.class
endif

.PHONY: shadowTestJar
shadowTestJar: $(SHADOW_TEST_JAR)

$(SHADOW_TEST_JAR): $(GRADLE_SHADOW_TEST_JAR) build-info.properties
	mkdir -p $(dir $(SHADOW_TEST_JAR))
	cp $(GRADLE_SHADOW_TEST_JAR) $(SHADOW_TEST_JAR)
	$(JAR) -uf $(SHADOW_TEST_JAR) build-info.properties

ifdef HAIL_COMPILE_NATIVES
$(GRADLE_SHADOW_TEST_JAR): native-lib-prebuilt
endif
ifdef HAIL_DEBUG_MODE
$(GRADLE_SHADOW_TEST_JAR): $(JAR_DEBUG_CLASSES)
endif
$(GRADLE_SHADOW_TEST_JAR): $(JAR_SOURCES) env/HAIL_DEBUG_MODE env/ELASTIC_MAJOR_VERSION
	./gradlew shadowTestJar $(GRADLE_ARGS)
ifdef HAIL_DEBUG_MODE
	$(JAR) -uf $(GRADLE_SHADOW_TEST_JAR) -C $(BUILD_DEBUG_PREFIX) is/hail/annotations/Memory.class
endif

jars: $(SHADOW_JAR) $(SHADOW_TEST_JAR)

$(NON_SHADOW_JAR): $(GRADLE_NON_SHADOW_JAR) build-info.properties
	mkdir -p $(dir $(NON_SHADOW_JAR))
	cp $(GRADLE_NON_SHADOW_JAR) $(NON_SHADOW_JAR)
	$(JAR) -uf $(NON_SHADOW_JAR) build-info.properties

ifdef HAIL_COMPILE_NATIVES
$(GRADLE_NON_SHADOW_JAR): native-lib-prebuilt
endif
ifdef HAIL_DEBUG_MODE
$(GRADLE_NON_SHADOW_JAR): $(JAR_DEBUG_CLASSES)
endif
$(GRADLE_NON_SHADOW_JAR): $(JAR_SOURCES) env/HAIL_DEBUG_MODE env/ELASTIC_MAJOR_VERSION
	./gradlew jar $(GRADLE_ARGS)
ifdef HAIL_DEBUG_MODE
	$(JAR) -uf $(GRADLE_NON_SHADOW_JAR) -C $(BUILD_DEBUG_PREFIX) is/hail/annotations/Memory.class
endif

.PHONY: jvm-test
ifdef HAIL_COMPILE_NATIVES
jvm-test: native-lib-prebuilt
endif
jvm-test: build-info.properties $(JAR_SOURCES) $(JAR_TEST_SOURCES)
	+./pgradle test $(GRADLE_ARGS) $(GRADLE_TEST_ARGS)

.PHONY: services-jvm-test
ifdef HAIL_COMPILE_NATIVES
services-jvm-test: native-lib-prebuilt
endif
services-jvm-test: build-info.properties $(JAR_SOURCES) $(JAR_TEST_SOURCES)
	./gradlew testServices $(GRADLE_ARGS) $(GRADLE_TEST_ARGS)

.PHONY: fs-jvm-test
ifdef HAIL_COMPILE_NATIVES
fs-jvm-test: native-lib-prebuilt
endif
fs-jvm-test: build-info.properties $(JAR_SOURCES) $(JAR_TEST_SOURCES) upload-qob-test-resources
	! [ -z $(HAIL_CLOUD) ]  # call like make fs-jvm-test HAIL_CLOUD=gcp or azure
	! [ -z $(NAMESPACE) ]  # call like make fs-jvm-test NAMEPSPACE=default
	HAIL_CLOUD=$(HAIL_CLOUD) \
	HAIL_DEFAULT_NAMESPACE=$(NAMESPACE) \
	HAIL_FS_TEST_CLOUD_RESOURCES_URI=$(CLOUD_HAIL_TEST_RESOURCES_DIR)fs \
	HAIL_TEST_STORAGE_URI=$(TEST_STORAGE_URI) \
	./gradlew testFS $(GRADLE_ARGS) $(GRADLE_TEST_ARGS)

# javac args from compileJava in build.gradle
$(BUILD_DEBUG_PREFIX)/%.class: src/debug/scala/%.java
	@mkdir -p $(BUILD_DEBUG_PREFIX)
	$(JAVAC) -d $(BUILD_DEBUG_PREFIX) -Xlint:all -Werror -XDenableSunApiLintControl -XDignore.symbol.file $<

build-info.properties: env/REVISION env/SHORT_REVISION env/BRANCH
build-info.properties: env/SPARK_VERSION env/HAIL_PIP_VERSION
build-info.properties: Makefile
	echo '[Build Metadata]' > $@
	echo 'user=$(USER)' >> $@
	echo 'revision=$(REVISION)' >> $@
	echo 'branch=$(BRANCH)' >> $@
	echo 'date=$(shell date -u +%Y-%m-%dT%H:%M:%SZ)' >> $@
	echo 'sparkVersion=$(SPARK_VERSION)' >> $@
	echo 'hailPipVersion=$(HAIL_PIP_VERSION)' >> $@

.PHONY: python-version-info
python-version-info: $(PYTHON_VERSION_INFO)

python/hail/hail_revision: env/REVISION
	echo $(REVISION) > $@

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

$(PYTHON_JAR): $(SHADOW_JAR)
	cp -f $(SHADOW_JAR) $@

$(FAST_PYTHON_JAR): $(NON_SHADOW_JAR)
	cp -f $(NON_SHADOW_JAR) $@

$(FAST_PYTHON_JAR_EXTRA_CLASSPATH): build.gradle
	./gradlew -q printClasspath > $(FAST_PYTHON_JAR_EXTRA_CLASSPATH)

.PHONY: python-jar
python-jar: $(PYTHON_JAR)

.PHONY: pytest
pytest: $(PYTHON_VERSION_INFO) $(INIT_SCRIPTS)
pytest: python/README.md $(FAST_PYTHON_JAR) $(FAST_PYTHON_JAR_EXTRA_CLASSPATH)
	cd python && $(HAIL_PYTHON3) -m pytest \
            -Werror:::hail -Werror:::hailtop -Werror::ResourceWarning \
            --log-cli-level=INFO \
            -s \
            -vv \
            -r A \
            --instafail \
            --durations=50 \
            --self-contained-html \
            --html=../build/reports/pytest.html \
            --timeout=120 \
            $(PYTEST_TARGET) \
            $(PYTEST_ARGS)

# NOTE: Look at upload-qob-test-resources target if test resources are missing
.PHONY: pytest-qob
pytest-qob: upload-qob-jar upload-qob-test-resources install-editable
	! [ -z $(NAMESPACE) ]  # call this like: make pytest-qob NAMESPACE=default
	cd python && \
  HAIL_QUERY_BACKEND=batch \
	HAIL_QUERY_JAR_URL=$$(cat ../upload-qob-jar) \
	HAIL_DEFAULT_NAMESPACE=$(NAMESPACE) \
	HAIL_TEST_RESOURCES_DIR='$(CLOUD_HAIL_TEST_RESOURCES_DIR)' \
	HAIL_DOCTEST_DATA_DIR='$(HAIL_DOCTEST_DATA_DIR)' \
	$(HAIL_PYTHON3) -m pytest \
            -Werror:::hail -Werror:::hailtop -Werror::ResourceWarning \
            --log-cli-level=INFO \
            -s \
            -vv \
            -r A \
            --instafail \
            --durations=50 \
            --self-contained-html \
            --html=../build/reports/pytest.html \
            --timeout=120 \
            $(PYTEST_TARGET) \
            $(PYTEST_ARGS)


.PHONY: doctest
doctest: $(PYTHON_VERSION_INFO) $(INIT_SCRIPTS)
doctest: python/README.md $(FAST_PYTHON_JAR) $(FAST_PYTHON_JAR_EXTRA_CLASSPATH)
	cd python && $(HAIL_PYTHON3) -m pytest \
            -Werror:::hail -Werror:::hailtop -Werror::ResourceWarning \
            --log-cli-level=INFO \
            -s \
            -vv \
            -r A \
            --instafail \
            --durations=50 \
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
            $(PYTEST_TARGET) \
            $(PYTEST_ARGS)


.PHONY: copy-py-files
copy-py-files: $(PYTHON_VERSION_INFO) $(INIT_SCRIPTS) $(PY_FILES) $(PYTHON_JAR)
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
	# Clear the bdist build cache before building the wheel
	cd build/deploy; rm -rf build; $(HAIL_PYTHON3) setup.py -q sdist bdist_wheel

# if the DEPLOY_REMOTE flag is not set, then deploy init scripts into a dev-username location
ifndef DEPLOY_REMOTE
DEV_CLARIFIER ?= $(shell whoami)-dev
cloud_base ?= gs://hail-30-day/hailctl/dataproc/$(DEV_CLARIFIER)/$(HAIL_VERSION)
UPLOAD_RETENTION =
else
cloud_base := gs://hail-common/hailctl/dataproc/$(HAIL_PIP_VERSION)
UPLOAD_RETENTION = gcloud storage objects update "$(cloud_base)/*" --temporary-hold
endif

wheel_cloud_path := $(cloud_base)/hail-$(HAIL_PIP_VERSION)-py3-none-any.whl
resources := $(wildcard python/hailtop/hailctl/dataproc/resources/*)
$(eval $(call ENV_VAR,cloud_base))
$(eval $(call ENV_VAR,wheel_cloud_path))


.PHONY: check-pip-lockfile
check-pip-lockfile:
	$(PIP) install 'pip-tools==6.13.0' && bash ../check_pip_requirements.sh python


python/hailtop/hailctl/deploy.yaml: env/cloud_base env/wheel_cloud_path
python/hailtop/hailctl/deploy.yaml: $(resources)
	rm -f $@
	echo "dataproc:" >> $@
	for FILE in $(notdir $(resources)); do \
	  echo "  $$FILE: $(cloud_base)/$$FILE" >> $@ || exit 1; done
	echo "  wheel: $(wheel_cloud_path)" >> $@
	printf "  pip_dependencies: " >> $@
	cat python/pinned-requirements.txt | sed '/^[[:blank:]]*#/d;s/#.*//' | grep -v pyspark | tr "\n" "|||" | tr -d '[:space:]' >> $@

.PHONY: upload-artifacts
upload-artifacts: $(WHEEL)
	gcloud storage cp $(resources) $(WHEEL) $(cloud_base)
	gcloud storage objects update -r $(cloud_base) --add-acl-grant=entity=AllUsers,role=READER
	$(UPLOAD_RETENTION)


# NOTE: 1-day expiration of the test bucket means that this
# target must be run at least once a day. To trigger this target to re-run,
# > rm upload-qob-test-resources
upload-qob-test-resources: $(shell git ls-files src/test/resources)
upload-qob-test-resources: $(shell git ls-files python/hail/docs/data)
	! [ -z $(NAMESPACE) ]  # call this like: make upload-qob-test-resources NAMESPACE=default
	gcloud storage cp src/test/resources/\* $(CLOUD_HAIL_TEST_RESOURCES_DIR)
	gcloud storage cp python/hail/docs/data/\* $(CLOUD_HAIL_DOCTEST_DATA_DIR)
	# # In Azure, use the following instead of gcloud storage cp
	# python3 -m hailtop.aiotools.copy -vvv 'null' '[\
        #      {"from":"src/test/resources","to":"$(CLOUD_HAIL_TEST_RESOURCES_DIR)"},\
        #      {"from":"python/hail/docs/data","to":"$(CLOUD_HAIL_DOCTEST_DATA_DIR)"}\
        # ]'
	# touch $@

# NOTE: 1-day expiration of the test bucket means that this
# target must be run at least once a day if using a dev NAMESPACE.
# To trigger this target to re-run,
# > rm upload-qob-jar
upload-qob-jar: $(SHADOW_JAR)
	! [ -z $(NAMESPACE) ]  # call this like: make upload-qob-jar NAMESPACE=default
	bash ./scripts/upload_qob_jar.sh $(NAMESPACE) $(REVISION) $(SHADOW_JAR) $@

# if the installation of Hail has changed, i.e. you pip-installed a non-development version of Hail,
# this file will be newer than "install-editable"
CURRENT_INSTALL = $(shell pip3 show -qq hail | grep Location: | sed 's/Location: //')

install-editable: $(if ($(CURRENT_INSTALL),),,$(CURRENT_INSTALL)/hail/__init__.py)
install-editable: python/setup.py
install-editable: python/requirements.txt
install-editable: python/README.md
install-editable: $(PYTHON_VERSION_INFO) $(INIT_SCRIPTS)
install-editable: $(FAST_PYTHON_JAR) $(FAST_PYTHON_JAR_EXTRA_CLASSPATH)
	-$(PIP) uninstall -y hail
	cd python && $(PIP) install -e .
	hailctl config set query/backend spark
	touch install-editable

.PHONY: install-for-qob
install-for-qob: upload-qob-jar install-editable
	! [ -z $(NAMESPACE) ]  # call this like: make install-for-qob NAMESPACE=default
	javac -version 2>&1 | grep -e '1\.8\.0'  # install-for-qob requires javac version 1.8, see https://discuss.hail.is/t/on-mac-os-x-how-do-i-install-and-use-java-8-if-i-already-have-a-different-version-of-java-installed/831/2
	hailctl config set query/backend batch
	hailctl config set query/jar_url $$(cat upload-qob-jar)
	hailctl dev config set default_namespace $(NAMESPACE)

.PHONY: install
install: $(WHEEL)
	-$(PIP) uninstall -y hail
	$(PIP) install $(WHEEL)
	hailctl config set query/backend spark

.PHONY: install-on-cluster
install-on-cluster: $(WHEEL) check-pip-lockfile
	sed '/^pyspark/d' python/pinned-requirements.txt | grep -v -e '^[[:space:]]*#' -e '^$$' | tr '\n' '\0' | xargs -0 $(PIP) install -U
	-$(PIP) uninstall -y hail
	$(PIP) install $(WHEEL) --no-deps
	hailctl config set query/backend spark

.PHONY: install-hailctl
install-hailctl: install upload-artifacts

.PHONY: test-dataproc-37
test-dataproc-37: install-hailctl
	bash scripts/test-dataproc.sh "GRCh37"

.PHONY: test-dataproc-38
test-dataproc-38: install-hailctl
	bash scripts/test-dataproc.sh "GRCh38"

# install skopeo
# use curl version >=7.55.0
#
# set DEPLOY_REMOTE to the hail-is/hail remote
#
# create an access token with full repo privileges at https://github.com/settings/tokens
# create a file that contains "Authorization: token YOUR_ACCESS_TOKEN_HERE"
# set GITHUB_OAUTH_HEADER_FILE to that filename
#
# create the hailgenetics/hail image (see /docker) and place it somewhere skopeo can access it
# set HAIL_GENETICS_HAIL_IMAGE to that image's full name including the protocol, e.g. docker://us-docker.pkg.dev/hail-vdc/hailgenetics/hail:1234abcd
#
# create the hailgenetics/hail image using the Python 3.10 ubuntu base image (see /docker) and place it somewhere skopeo can access it
# set HAIL_GENETICS_HAIL_IMAGE_PY_3_10 to that image's full name including the protocol, e.g. docker://us-docker.pkg.dev/hail-vdc/hailgenetics/hail:1234abcd-py3.10
#
# create the hailgenetics/hail image using the Python 3.11 ubuntu base image (see /docker) and place it somewhere skopeo can access it
# set HAIL_GENETICS_HAIL_IMAGE_PY_3_11 to that image's full name including the protocol, e.g. docker://us-docker.pkg.dev/hail-vdc/hailgenetics/hail:1234abcd-py3.11
#
# create the hailgenetics/hailtop image (see /docker) and place it somewhere skopeo can access it
# set HAIL_GENETICS_HAILTOP_IMAGE to that image's full name including the protocol, e.g. docker://us-docker.pkg.dev/hail-vdc/hailgenetics/hailtop:1234abcd
#
# create the hailgenetics/vep-grch37-85 image (see /docker) and place it somewhere skopeo can access it
# set HAIL_GENETICS_VEP_GRCH37_85_IMAGE to that image's full name including the protocol, e.g. docker://us-docker.pkg.dev/hail-vdc/hailgenetics/vep-grch37-85:1234abcd
#
# create the hailgenetics/vep-grch38-95 image (see /docker) and place it somewhere skopeo can access it
# set HAIL_GENETICS_VEP_GRCH38_95_IMAGE to that image's full name including the protocol, e.g. docker://us-docker.pkg.dev/hail-vdc/hailgenetics/vep-grch38-95:1234abcd
#
# cluster to find the correct Scala and Spark versions because the version webpage does not include
# sufficient detail)
# set WHEEL_FOR_AZURE to the path to that wheel file
#
# Create a hail docs www.tar.gz file (see build.yaml "make_docs" rule) and set WEBSITE_TAR to the path to www.tar.gz
#
.PHONY: release
release: test-dataproc $(WHEEL)
	bash scripts/release.sh $(HAIL_PIP_VERSION) $(HAIL_VERSION) $(REVISION) $(DEPLOY_REMOTE) $(WHEEL) $(GITHUB_OAUTH_HEADER_FILE) $(HAIL_GENETICS_HAIL_IMAGE) $(HAIL_GENETICS_HAIL_IMAGE_PY_3_10) $(HAIL_GENETICS_HAIL_IMAGE_PY_3_11) $(HAIL_GENETICS_HAILTOP_IMAGE) $(HAIL_GENETICS_VEP_GRCH37_85_IMAGE) $(HAIL_GENETICS_VEP_GRCH38_95_IMAGE) $(WHEEL_FOR_AZURE) $(WEBSITE_TAR) $(GCP_PROJECT) $(HAIL_HAIL_DIR)/../infra/gcp-broad/gcp-ar-cleanup-policy.txt

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
	      BUILDDIR=$(HAIL_HAIL_DIR)/build/docs/batch \
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
	        BUILDDIR=$(HAIL_HAIL_DIR)/build/docs/hail \
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
hail-docs-no-test: install-editable
hail-docs-no-test: $(PYTHON_VERSION_INFO) python/hail/docs/change_log.rst
	$(MAKE) -C python/hail/docs \
	        BUILDDIR=$(HAIL_HAIL_DIR)/build/docs/hail \
	        html
	mkdir -p build/www/docs
	rm -rf build/www/docs/$(HAIL_MAJOR_MINOR_VERSION)
	cp -R build/docs/hail/html build/www/docs/$(HAIL_MAJOR_MINOR_VERSION)
	find build/www/docs/$(HAIL_MAJOR_MINOR_VERSION) \
	     -iname *.html \
	     -type f \
	     -exec $(SED_INPLACE) -e "s/\.css/\.css\?v\=$(HAIL_CACHE_VERSION)/" {} +;
	@echo Built docs: file://$(HAIL_HAIL_DIR)/build/www/docs/$(HAIL_MAJOR_MINOR_VERSION)/index.html

.PHONY: upload-docs
upload-docs: hail-docs batch-docs
	cd build && tar czf www.tar.gz www
	gcloud storage cp build/www.tar.gz $(docs_location)
	gcloud storage objects update -r $(docs_location) --temporary-hold
	gcloud storage objects update -r $(docs_location) --add-acl-grant=entity=AllUsers,role=READER

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
	rm -rf $(FAST_PYTHON_JAR)
	rm -rf $(FAST_PYTHON_JAR_EXTRA_CLASSPATH)
	rm -rf python/README.md
	rm -rf build-info.properties
	rm -rf $(PYTHON_VERSION_INFO)
	rm -rf python/hail/docs/change_log.rst

.PHONY: update-hail-repl
update-hail-repl: NAMESPACE ?= default
update-hail-repl: wheel
	kubectl -n $(NAMESPACE) cp $(WHEEL) $$(kubectl get pods -n $(NAMESPACE) -l app=hail-repl | tail -n +2 | awk '{print $$1}'):.
	kubectl -n $(NAMESPACE) exec -it $$(kubectl get pods -n $(NAMESPACE) -l app=hail-repl | tail -n +2 | awk '{print $$1}') -- pip3 install -U hail-$(HAIL_PIP_VERSION)-py3-none-any.whl
