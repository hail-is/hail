include env_var.mk

MAKEFLAGS += --no-builtin-rules
.SUFFIXES:

PARALLELISM ?= $(shell python -c 'import multiprocessing; print(multiprocessing.cpu_count())')
SPARK_VERSION ?= 2.2.0
# Hail has three notions of version:
#  - hail short version: MAJOR.MINOR
#  - hail pip version: MAJOR.MINOR.PATCH
#  - hail version: MAJOR.MINOR.PATCH-GIT_SHA, calculated by generateBuildInfo
MAJOR ?= 0
MINOR ?= 2
PATCH ?= 10
HAIL_PIP_VERSION := $(MAJOR).$(MINOR).$(PATCH)$(PIP_VERSION_SUFFIX)
GRADLE_ARGS := $(GRADLE_ARGS) -Dspark.version=$(SPARK_VERSION)
GIT_SHA := $(shell git rev-parse HEAD)

$(eval $(call ENV_VAR,SPARK_VERSION))
$(eval $(call ENV_VAR,HAIL_PIP_VERSION))
$(eval $(call ENV_VAR,GRADLE_ARGS))
$(eval $(call ENV_VAR,GIT_SHA))

IGNORED_FILE_PATTERN := \( -name '*~' \)

build/dev-conda: python/dev-environment.yml
	conda env update -f $< || conda env create -f $<
	mkdir -p build
	touch build/dev-conda

build/native-lib: $(shell find src/main/c -type f -not $(IGNORED_FILE_PATTERN))
	cd src/main/c && $(MAKE)
	mkdir -p build
	touch build/native-lib

.PHONY: native-lib
native-lib: build/native-lib

build/test-native-lib: $(shell find src/main/c -type f -not $(IGNORED_FILE_PATTERN))
	cd src/main/c && $(MAKE) test
	mkdir -p build/test-native-lib
	touch build/test-native-lib

.PHONY: test-native-lib
test-native-lib: build/test-native-lib

build/prebuilt-native-lib: $(shell find src/main/c -type f -not $(IGNORED_FILE_PATTERN))
	cd src/main/c && $(MAKE) prebuilt
	mkdir -p build/prebuilt-native-lib
	touch build/prebuilt-native-lib

.PHONY: prebuilt-native-lib
prebuilt-native-lib: build/prebuilt-native-lib

JVM_FILES := $(shell find src/main/java src/main/scala src/main/resources \( -not $(IGNORED_FILE_PATTERN) \))

build/compile-scala: src/main/resources/build-info.properties env/GRADLE_ARGS
	./gradlew $(GRADLE_ARGS) compileScala
	mkdir -p build
	touch build/compile-scala

build/libs/hail-all-spark.jar: src/main/resources/build-info.properties
build/libs/hail-all-spark.jar: build.gradle settings.gradle $(JVM_FILES)
build/libs/hail-all-spark.jar: env/GRADLE_ARGS
	$(info changed files: $?)
	./gradlew $(GRADLE_ARGS) shadowJar

.PHONY: compile-scala
compile-scala: build/compile-scala

.PHONY: jar
jar: build/libs/hail-all-spark.jar

build/distributions/hail-python.zip: $(shell find python -type f) env/GRADLE_ARGS
	./gradlew $(GRADLE_ARGS) archiveZip

.PHONY: zip
zip: build/distributions/hail-python.zip

IGNORED_PY_FILE_PATTERN = \( $(IGNORED_FILE_PATTERN) -o -name '*.pyc' \)
PY_FILES := $(shell find python/hail -type f -not $(IGNORED_PY_FILE_PATTERN))
PY_TEST_FILES := $(shell find python/tests -type f -not $(IGNORED_PY_FILE_PATTERN))
PYTHON_PACKAGE_JAR_LOCATION := python/hail/hail-all-spark.jar

build/pip-prepare: build/libs/hail-all-spark.jar $(PY_FILES) env/SPARK_VERSION
build/pip-prepare: python/README.md python/hail/hail_pip_version
build/pip-prepare: python/hail/hail_version
	cp build/libs/hail-all-spark.jar $(PYTHON_PACKAGE_JAR_LOCATION)
	printf $(SPARK_VERSION) > python/spark_version
	mkdir -p build
	touch build/pip-prepare

.PHONY: pip-deploy
pip-deploy: build/pip-prepare build/dev-conda
	. ../loadconda && conda activate hail && ./python/deploy.sh

.PHONY: pip-install
pip-install: build/pip-install

build/pip-install: python/setup.py python/MANIFEST.in python/setup.cfg build/pip-prepare build/dev-conda
	. ../loadconda && conda activate hail && pip install -U ./python
	mkdir -p build
	touch build/pip-install

.PHONY: test-python
test-python: build/test-python

build/test-python: build/pip-install build/dev-conda $(PY_TEST_FILES)
# re: PY_IGNORE_IMPORTMISMATCH, https://github.com/pytest-dev/pytest/issues/2042#issuecomment-429289164
	. ../loadconda && conda activate hail && PY_IGNORE_IMPORTMISMATCH=1 pytest -v \
	  -n $(PARALLELISM) \
	  --dist=loadscope \
	  --noconftest \
	  --color=yes \
	  -r a \
	  --html=build/reports/pytest.html \
	  --self-contained-html \
	  --doctest-modules \
	  --doctest-glob='*.rst' \
	  --ignore=python/hail/docs/conf.py \
	  --ignore=python/hail/docs/doctest_write_data.py \
	  python/tests python/src $(PYTEST_ARGS)
	mkdir -p build
	touch build/test-python

.PHONY: benchmark
benchmark: build/pip-install build/dev-conda
	. ../loadconda && conda activate hail && cd python && python -m benchmark

.PHONY: gradle-tests
gradle-test: env/GRADLE_ARGS
	./gradlew $(GRADLE_ARGS) test

.PHONY: gradle-tests
gradle-test-cpp-codegen: env/GRADLE_ARGS
	./gradlew $(GRADLE_ARGS) testCppCodegen

build/credentials.json:
	mkdir -p build
	scp gold.broadinstitute.org:/psych/genetics_data/working/cseed/hail-internal/credentials.json $@

build/twine-credentials/pypi-username: build/credentials.json
	mkdir -p build/twine-credentials
	jq -r '.pypi.username' $< > $@

build/twine-credentials/pypi-password: build/credentials.json
	mkdir -p build/twine-credentials
	jq -r '.pypi.password' $< > $@

.PHONY: test-pip-deploy
test-pip-deploy: build/twine-credentials/pypi-username build/twine-credentials/pypi-password
	PIP_VERSION_SUFFIX=".dev$(shell curl -sSL https://pypi.python.org/pypi/hail/json | python next_unused_dev_pypi_version.py)" \
	  HAIL_TWINE_CREDS_FOLDER=build/twine-credentials \
	  $(MAKE) pip-deploy
	mkdir -p build
	touch build/test-pip-deploy

.PHONY: test
test: test-python gradle-test gradle-test-cpp-codegen build/test-native-lib
test: test-pip-install

src/main/resources/build-info.properties: env/SPARK_VERSION env/HAIL_PIP_VERSION env/GIT_SHA
	bash generate-build-info.sh $(SPARK_VERSION) $(HAIL_PIP_VERSION) $(GIT_SHA)

python/hail/hail_pip_version: env/HAIL_PIP_VERSION
	printf $(HAIL_PIP_VERSION) > $@

python/hail/hail_version: env/HAIL_PIP_VERSION env/GIT_SHA
	printf $(HAIL_PIP_VERSION)-$(shell printf "$(GIT_SHA)" | head -c 12) > $@

python/README.md: ../README.md
	cp $< python/

.PHONY: docs
docs: SPHINXOPTS = -tchecktutorial
docs: docs-no-test
	PYSPARK_SUBMIT_ARGS='--master local[4] pyspark-shell' SPHINXOPTS='-tchecktutorial' bash python/hail/docs/makeDocs.sh

.PHONY: docs-no-test
docs-no-test: build/libs/hail-all-spark.jar python/hail/hail_pip_version python/hail/hail_version
	PYSPARK_SUBMIT_ARGS='--master local[4] pyspark-shell' SPHINXOPTS=$(SPHINXOPTS) bash python/hail/docs/makeDocs.sh

.PHONY: clean-docs
clean-docs:
	rm -rf build/www build/tmp/python build/tmp/docs

.PHONY: clean
clean: clean-docs clean-env
	./gradlew $(GRADLE_ARGS) clean
	cd src/main/c && %(MAKE) clean
	rm -rf $(PYTHON_PACKAGE_JAR_LOCATION)

test-apiserver: jar
	./test-apiserver.sh
