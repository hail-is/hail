.PHONY: shadowJar build-info jars clean \
  src/main/resources/build-info.properties \
  python/hail/hail_version python/hail/hail_pip_version


REVISION := $(shell git rev-parse HEAD)
SHORT_REVISION := $(shell git rev-parse --short=12 HEAD)
DATE := $(shell date -u +%Y-%m-%dT%H:%M:%SZ)
BRANCH := $(shell git rev-parse --abbrev-ref HEAD)
URL := $(shell git config --get remote.origin.url)
SPARK_VERSION := 2.4.0
HAIL_PIP_VERSION := 0.2.16
PARALLELISM := 2

HAIL_PYTHON3 ?= python3
PIP ?= $(HAIL_PYTHON3) -m pip


ifdef HAIL_COMPILE_NATIVES
shadowJar:
	./gradlew releaseJar
else
shadowJar:
	./gradlew shadowJar
endif

build-info: src/main/resources/build-info.properties python/hail/hail_version python/hail/hail_pip_version \
	python/hailtop/hail_version python/hailtop/hail_pip_version

define properties
endef

src/main/resources/build-info.properties:
	echo '[Build Metadata]' > $@
	echo 'user=$(USER)' >> $@
	echo 'revision=$(REVISION)' >> $@
	echo 'branch=$(BRANCH)' >> $@
	echo 'date=$(DATE)' >> $@
	echo 'url=$(URL)' >> $@
	echo 'sparkVersion=$(SPARK_VERSION)' >> $@
	echo 'hailPipVersion=$(HAIL_PIP_VERSION)' >> $@

python/hail/hail_version:
	echo $(HAIL_PIP_VERSION)-$(SHORT_REVISION) > python/hail/hail_version

python/hailtop/hail_version: python/hail/hail_version
	cp -f $< $@

python/hail/hail_pip_version:
	echo $(HAIL_PIP_VERSION) > python/hail/hail_pip_version

python/hailtop/hail_pip_version: python/hail/hail_pip_version
	cp -f $< $@

jars: build-info
	./gradlew shadowTestJar shadowJar

python/README.md: ../README.md
	cp ../README.md python/

python/hail/hail-all-spark.jar: shadowJar
	cp build/libs/hail-all-spark.jar python/hail/

.PHONY: install-editable
install-editable: build-info init-scripts
install-editable: python/README.md python/hail/hail-all-spark.jar
	cd python && $(PIP) install -e

.PHONY: pytest
pytest: build-info shadowJar init-scripts
pytest: python/README.md python/hail/hail-all-spark.jar
	cd python && $(HAIL_PYTHON3) setup.py pytest \
    --addopts '-v -n $(PARALLELISM) --dist=loadscope --noconftest --color=no -r a --html=build/reports/pytest.html --self-contained-html $(PYTEST_ARGS)'

.PHONY: wheel
wheel: build-info shadowJar init-scripts
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
else
CLOUD_SUB_FOLDER := $(HAIL_PIP_VERSION)
endif

HAILCTL_BUCKET_BASE ?= gs://hail-common/hailctl/dataproc

cloud_base := $(HAILCTL_BUCKET_BASE)/$(DEV_CLARIFIER)$(CLOUD_SUB_FOLDER)
wheel_cloud_path := $(cloud_base)/hail-$(HAIL_PIP_VERSION)-py3-none-any.whl
wheel_path := build/deploy/dist/hail-$(HAIL_PIP_VERSION)-py3-none-any.whl
resources := $(wildcard python/hailtop/hailctl/dataproc/resources/*)
.PHONY: python/hailtop/hailctl/deploy.yaml
python/hailtop/hailctl/deploy.yaml:
	rm -f $@
	echo "dataproc:" >> $@
	for FILE in $(notdir $(resources)); do \
	  echo "  $$FILE: $(cloud_base)/$$FILE" >> $@ || exit 1; done
	echo "  wheel: $(wheel_cloud_path)" >> $@
	echo "  pip_dependencies: $(shell cat python/requirements.txt | grep -v pyspark | tr "\n" "|||")" >> $@

.PHONY: upload-artifacts
upload-artifacts: wheel
	gsutil -m cp -r $(resources) $(wheel_path) $(cloud_base)
	gsutil -m acl set -r public-read $(cloud_base)

.PHONY: install
install: wheel
	-$(PIP) uninstall -y hail
	$(PIP) install $(wheel_path)

.PHONY: install-hailctl
install-hailctl: install upload-artifacts


cluster_name := cluster-$(shell whoami)-$(shell echo $$RANDOM)
.PHONY: test-dataproc
test-dataproc: install-hailctl
	hailctl dataproc start $(cluster_name) --max-idle 10m --vep GRCh37
	for FILE in `ls python/cluster-tests`; do \
	  hailctl dataproc submit $(cluster_name) python/cluster-tests/$$FILE || exit 1; done || exit

.PHONY: init-scripts
init-scripts: python/hailtop/hailctl/deploy.yaml

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

clean:
	./gradlew clean
	rm -rf build/deploy
	rm -rf python/hail/hail-all-spark.jar
	rm -rf python/README.md
