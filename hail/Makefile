.PHONY: shadowJar build-info jars clean \
  src/main/resources/build-info.properties \
  python/hail/hail_version python/hail/hail_pip_version


REVISION := $(shell git rev-parse HEAD)
SHORT_REVISION := $(shell git rev-parse --short=12 HEAD)
DATE := $(shell date -u +%Y-%m-%dT%H:%M:%SZ)
BRANCH := $(shell git rev-parse --abbrev-ref HEAD)
URL := $(shell git config --get remote.origin.url)
SPARK_VERSION := 2.4.0
HAIL_PIP_VERSION := 0.2.15

HAIL_PYTHON3 ?= python3
PIP ?= $(HAIL_PYTHON3) -m pip


shadowJar:
	./gradlew shadowJar

releaseJar:
	./gradlew releaseJar

build-info: src/main/resources/build-info.properties python/hail/hail_version python/hail/hail_pip_version \
	python/hailctl/hail_version python/hailctl/hail_pip_version

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

python/hailctl/hail_version: python/hail/hail_version
	cp -f $< $@

python/hail/hail_pip_version:
	echo $(HAIL_PIP_VERSION) > python/hail/hail_pip_version

python/hailctl/hail_pip_version: python/hail/hail_pip_version
	cp -f $< $@

jars: build-info
	./gradlew shadowTestJar shadowJar

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
DEV_CLARIFIER := $(shell whoami)-dev
CLOUD_SUB_FOLDER := $(HAIL_PIP_VERSION)-$(SHORT_REVISION)
else
CLOUD_SUB_FOLDER := $(HAIL_PIP_VERSION)
endif

HAILCTL_BUCKET_BASE ?= gs://hail-common/hailctl/dataproc

cloud_base := $(HAILCTL_BUCKET_BASE)/$(DEV_CLARIFIER)/$(CLOUD_SUB_FOLDER)
wheel_cloud_path := $(cloud_base)/hail-$(HAIL_PIP_VERSION)-py3-none-any.whl
wheel_path := build/deploy/dist/hail-$(HAIL_PIP_VERSION)-py3-none-any.whl
resources := $(wildcard python/hailctl/dataproc/resources/*)
.PHONY: python/hailctl/deploy.yaml
python/hailctl/deploy.yaml:
	rm -f python/hailctl/deploy.yaml
	echo "dataproc:" >> python/hailctl/deploy.yaml
	for FILE in $(notdir $(resources)); do \
	  echo "  $$FILE: $(cloud_base)/$$FILE" >> python/hailctl/deploy.yaml || break; done || exit
	echo "  wheel: $(wheel_cloud_path)" >> python/hailctl/deploy.yaml
	echo "  pip_dependencies: $(shell cat python/requirements.txt | grep -v pyspark | tr "\n" "|||")" >> python/hailctl/deploy.yaml

.PHONY: upload-artifacts
upload-artifacts: wheel
	gsutil -m cp -r $(resources) $(wheel_path) $(cloud_base)
	gsutil -m acl set -r public-read $(cloud_base)

.PHONY: install-wheel
install-wheel: wheel
	-$(PIP) uninstall -y hail
	$(PIP) install $(wheel_path)

.PHONY: install-hailctl-wheel
install-hailctl-wheel: install-wheel upload-artifacts


cluster_name := cluster-$(shell whoami)-$(shell echo $$RANDOM)
.PHONY: test-dataproc
test-dataproc: install-hailctl-wheel
	hailctl dataproc start $(cluster_name) --max-idle 10m --vep GRCh37
	for FILE in `ls python/cluster-tests`; do \
	  hailctl dataproc submit $(cluster_name) python/cluster-tests/$$FILE || break; done || exit

.PHONY: init-scripts
init-scripts: python/hailctl/deploy.yaml

DEPLOYED_VERSION = $(shell \
  $(pip) --no-cache-dir search hail \
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
pypi-deploy: export TWINE_USERNAME := $(shell cat $(HAIL_TWINE_CREDS_FOLDER)/pypi-username)
pypi-deploy: export TWINE_PASSWORD := $(shell cat $(HAIL_TWINE_CREDS_FOLDER)/pypi-password)
pypi-deploy: check-pypi wheel
	twine upload build/deploy/dist/*

TAG_EXISTS = $(shell git ls-remote --exit-code --tags origin $(HAIL_PIP_VERSION) || echo "does not exist")
.PHONY: check-tag
check-tag:
	if [ -n "$(TAG_EXISTS)" ]; then echo "tag $(HAIL_PIP_VERSION) already exists"; exit 1; fi

.PHONY: tag
tag: check-tag
	git tag $(HAIL_PIP_VERSION) -m "Hail version $(HAIL_PIP_VERSION)"
	git push https://github.com/hail-is/hail.git $(HAIL_PIP_VERSION)

docs_location := gs://hail-common/builds/0.2/docs/hail-0.2-docs-$(REVISION).tar.gz
local_sha_location := build/deploy/latest-hash-spark-$(SPARK_VERSION).txt
cloud_sha_location := gs://hail-common/builds/0.2/latest-hash/cloudtools-5-spark-2.4.0.txt
.PHONY: set-docs-sha
set-docs-sha:
	gsutil ls $(docs_location)  # make sure file exists
	echo "$(REVISION)" > $(local_sha_location)
	gsutil cp $(local_sha_location) $(cloud_sha_location)
	gsutil acl set public-read $(cloud_sha_location)

.PHONY: deploy
deploy: pypi-deploy set-docs-sha tag

clean:
	./gradlew clean
	rm -rf build/deploy
