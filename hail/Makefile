.PHONY: shadowJar build-info jars clean \
  src/main/resources/build-info.properties python/hail/hail_version python/hail/hail_pip_version

REVISION=$(shell git rev-parse HEAD)
SHORT_REVISION=$(shell git rev-parse --short=12 HEAD)
DATE=$(shell date -u +%Y-%m-%dT%H:%M:%SZ)
BRANCH=$(shell git rev-parse --abbrev-ref HEAD)
URL=$(shell git config --get remote.origin.url)
SPARK_VERSION=2.4.0
HAIL_PIP_VERSION=0.2.14

shadowJar:
	./gradlew shadowJar

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
	make -C src/main/c
	mvn -B compile test-compile jar:jar jar:test-jar assembly:single

clean:
	make -C src/main/c clean
	mvn -B clean
