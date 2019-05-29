.PHONY: shadowJar build-info jars wheel install-wheel clean \
  src/main/resources/build-info.properties \
  python/hail/hail_version python/hail/hail_pip_version \
  ../hailctl/python/hailctl/deploy.yaml

REVISION=$(shell git rev-parse HEAD)
SHORT_REVISION=$(shell git rev-parse --short=12 HEAD)
DATE=$(shell date -u +%Y-%m-%dT%H:%M:%SZ)
BRANCH=$(shell git rev-parse --abbrev-ref HEAD)
URL=$(shell git config --get remote.origin.url)
SPARK_VERSION=2.4.0
HAIL_PIP_VERSION=0.2.14

HAIL_PYTHON3?=python3

shadowJar:
	./gradlew shadowJar

releaseJar:
	./gradlew releaseJar

build-info: src/main/resources/build-info.properties python/hail/hail_version python/hail/hail_pip_version

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

python/hail/hail_pip_version:
	echo $(HAIL_PIP_VERSION) > python/hail/hail_pip_version

jars: build-info
	./gradlew shadowTestJar shadowJar

wheel: build-info init-scripts shadowJar
	rm -rf build/deploy
	mkdir -p build/deploy
	mkdir -p build/deploy/src
	cp ../README.md build/deploy/
	cp deploy_scripts/setup.py build/deploy/
	cp deploy_scripts/setup.cfg build/deploy/
	rsync -rv \
	    --exclude '__pycache__/' \
	    --exclude 'docs/' \
	    --exclude '*.log' \
	    python/hail build/deploy/src/

	rsync -rv \
	    --exclude '__pycache__/' \
	    --exclude '*.log' \
	    ../hailctl/python/hailctl build/deploy/src/

	cp build/libs/hail-all-spark.jar build/deploy/src/hail/
	cp build/deploy/src/hail/hail_pip_version build/deploy/src/hailctl/hail_pip_version
	cp build/deploy/src/hail/hail_version build/deploy/src/hailctl/hail_version
	cd build/deploy; $(HAIL_PYTHON3) setup.py sdist bdist_wheel

install-wheel: wheel
	pip uninstall -y hail
	pip install build/deploy/dist/*.whl

# if the DEPLOY_PROD flag is not set, then deploy init scripts into a dev-username location
ifndef DEPLOY_PROD
DEV_SALT := $(shell whoami)-dev
endif

CLOUD_LOC = gs://hail-common/hailctl/dataproc/$(DEV_SALT)/$(HAIL_PIP_VERSION)/$(SHORT_REVISION)/$(FILENAME)
RESOURCES_TO_COPY = $(shell ls ../hailctl/python/hailctl/dataproc/resources)
../hailctl/python/hailctl/deploy.yaml:
	rm -f ../hailctl/python/hailctl/deploy.yaml
	echo "dataproc:" >> ../hailctl/python/hailctl/deploy.yaml
	$(foreach FILENAME, $(RESOURCES_TO_COPY), \
	    gsutil cp ../hailctl/python/hailctl/dataproc/resources/$(FILENAME) $(CLOUD_LOC); \
	    gsutil acl set public-read $(CLOUD_LOC); \
	    echo "  $(FILENAME): $(CLOUD_LOC)" >> ../hailctl/python/hailctl/deploy.yaml;)

.PHONY: pypi-deploy
init-scripts: ../hailctl/python/hailctl/deploy.yaml

DEPLOYED_VERSION=$(shell \
  pip --no-cache-dir search hail \
   | grep '^hail ' \
   | sed 's/hail (//' \
   | sed 's/).*//')
.PHONY: check-pypi
check-pypi:
	if [ "$(DEPLOYED_VERSION)" == "$(HAIL_PIP_VERSION)" ]; then \
	  echo "version $(HAIL_PIP_VERSION) already deployed"; exit 1; fi

$HAIL_TWINE_CREDS_FOLDER ?= /secrets/
TWINE_USERNAME=$(cat $(HAIL_TWINE_CREDS_FOLDER)/pypi-username)
TWINE_PASSWORD=$(cat $(HAIL_TWINE_CREDS_FOLDER)/pypi-password)

.PHONY: pypi-deploy
pypi-deploy: \
  TWINE_USERNAME=$(cat $(HAIL_TWINE_CREDS_FOLDER)/pypi-username) \
  TWINE_PASSWORD=$(cat $(HAIL_TWINE_CREDS_FOLDER)/pypi-password)
pypi-deploy: check-pypi wheel
	twine upload build/deploy/dist/*

TAG_EXISTS=$(shell git ls-remote --exit-code --tags origin $(HAIL_PIP_VERSION) || echo "does not exist")
.PHONY: check-tag
check-tag:
	if [ -n "$(TAG_EXISTS)" ]; then echo "tag $(HAIL_PIP_VERSION) already exists"; exit 1; fi

.PHONY: tag
tag: check-tag
	git tag $(HAIL_PIP_VERSION) -m "Hail version $(HAIL_PIP_VERSION)"
	git push https://github.com/hail-is/hail.git $(HAIL_PIP_VERSION)

.PHONY: deploy
deploy: pypi-deploy tag

clean:
	./gradlew clean
	rm -rf build/deploy