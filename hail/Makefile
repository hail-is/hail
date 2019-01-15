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
PATCH ?= 8
HAIL_PIP_VERSION := $(MAJOR).$(MINOR).$(PATCH)$(PIP_VERSION_SUFFIX)
GRADLE_ARGS := $(GRADLE_ARGS) -Dspark.version=$(SPARK_VERSION)
GIT_SHA := $(shell git rev-parse HEAD)
GIT_SHORT_SHA := $(shell printf "$(GIT_SHA)" | head -c 12)

$(eval $(call ENV_VAR,SPARK_VERSION))
$(eval $(call ENV_VAR,HAIL_PIP_VERSION))
$(eval $(call ENV_VAR,GRADLE_ARGS))
$(eval $(call ENV_VAR,GIT_SHA))
$(eval $(call ENV_VAR,GIT_SHORT_SHA))

build/dev-conda: python/dev-environment.yml
	conda env update -f $< || conda env create -f $<
	mkdir -p build
	touch build/dev-conda

.PHONY: native-lib
native-lib:
	cd src/main/c && $(MAKE)

.PHONY: test-native-lib
test-native-lib:
	cd src/main/c && $(MAKE) test

.PHONY: prebuilt-native-lib
prebuilt-native-lib:
	cd src/main/c && $(MAKE) prebuilt

.PHONY: compile-scala
compile-scala: src/main/resources/build-info.properties native-lib env/GRADLE_ARGS
	./gradlew $(GRADLE_ARGS) compileScala

JVM_FILES := $(shell find src \( -name '*.java' -o -name '*.scala' \))
JVM_RESOURCE_FILES := $(shell find src/main/resources src/test/resources)

build/libs/hail-all-spark.jar: src/main/resources/build-info.properties
build/libs/hail-all-spark.jar: build.gradle settings.gradle $(JVM_FILES)
build/libs/hail-all-spark.jar: $(JVM_RESOURCE_FILES) env/GRADLE_ARGS
	$(info changed files: $?)
	./gradlew $(GRADLE_ARGS) shadowJar

.PHONY: jar
jar: build/libs/hail-all-spark.jar

.PHONY: zip
zip: python/src/hail/hail_pip_version python/src/hail/hail_version env/GRADLE_ARGS
	./gradlew $(GRADLE_ARGS) archiveZip

IGNORED_FILE_PATTERN = \( -name '*~'  -o -name '*.pyc' \)
PY_FILES := $(shell find python/src/hail -type f -not $(IGNORED_FILE_PATTERN))
PY_TEST_FILES := $(shell find python/tests -type f -not $(IGNORED_FILE_PATTERN))
PYTHON_PACKAGE_JAR_LOCATION := python/src/hail/hail-all-spark.jar

build/pip-prepare: build/libs/hail-all-spark.jar $(PY_FILES) env/SPARK_VERSION
	cp $< $(PYTHON_PACKAGE_JAR_LOCATION)
	cp ../README.md python/
	printf $(SPARK_VERSION) > python/spark_version
	mkdir -p build
	touch build/pip-prepare

.PHONY: pip-deploy
pip-deploy: build/pip-prepare python/src/hail/hail_pip_version python/src/hail/hail_version build/dev-conda
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
	  --ignore=python/src/hail/docs/conf.py \
	  --ignore=python/src/hail/docs/doctest_write_data.py \
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

.PHONY: test
test: test-python gradle-test gradle-test-cpp-codegen test-native-lib

src/main/resources/build-info.properties: env/SPARK_VERSION env/HAIL_PIP_VERSION env/GIT_SHA
	bash generate-build-info.sh $(SPARK_VERSION) $(HAIL_PIP_VERSION) $(GIT_SHA)

python/src/hail/hail_pip_version: env/HAIL_PIP_VERSION
	printf $(HAIL_PIP_VERSION) > $@

python/src/hail/hail_version: env/HAIL_PIP_VERSION env/GIT_SHORT_SHA
	printf $(HAIL_PIP_VERSION)-$(GIT_SHORT_SHA) > $@

.PHONY: docs
docs: SPHINXOPTS = -tchecktutorial
docs: docs-no-test
	PYSPARK_SUBMIT_ARGS='--master local[4] pyspark-shell' SPHINXOPTS='-tchecktutorial' bash python/hail/docs/makeDocs.sh

.PHONY: docs-no-test
docs-no-test: build/libs/hail-all-spark.jar python/src/hail/hail_pip_version python/src/hail/hail_version
	PYSPARK_SUBMIT_ARGS='--master local[4] pyspark-shell' SPHINXOPTS=$(SPHINXOPTS) bash python/hail/docs/makeDocs.sh

.PHONY: clean-docs
clean-docs:
	rm -rf build/www build/tmp/python build/tmp/docs

.PHONY: clean
clean: clean-docs clean-env
	./gradlew $(GRADLE_ARGS) clean
	cd src/main/c && %(MAKE) clean
	rm -rf $(PYTHON_PACKAGE_JAR_LOCATION)
