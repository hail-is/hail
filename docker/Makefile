include ../config.mk

.PHONY: build
build: base service-base service-java-run-base hail-ubuntu

TOKEN = $(shell cat /dev/urandom | LC_ALL=C tr -dc 'a-z0-9' | head -c 12)

HAIL_UBUNTU_IMAGE := $(DOCKER_PREFIX)/hail-ubuntu:$(TOKEN)
BASE_IMAGE := $(DOCKER_PREFIX)/base:$(TOKEN)
SERVICE_BASE_IMAGE := $(DOCKER_PREFIX)/service-base:$(TOKEN)
SERVICE_JAVA_RUN_BASE_IMAGE := $(DOCKER_PREFIX)/service-java-run-base:$(TOKEN)
HAIL_PUBLIC_IMAGE := $(DOCKER_PREFIX)/hail-public:$(TOKEN)
GENETICS_PUBLIC_IMAGE := $(DOCKER_PREFIX)/genetics-public:$(TOKEN)

.PHONY: hail-ubuntu hail-ubuntu-image-ref
hail-ubuntu: hail-ubuntu-image-ref
hail-ubuntu-image-ref: hail-ubuntu/Dockerfile hail-ubuntu/hail-pip-install hail-ubuntu/pip.conf hail-ubuntu/hail-apt-get-install
	python3 ../ci/jinja2_render.py '{"global":{"docker_prefix":"$(DOCKER_PREFIX)"}}' hail-ubuntu/Dockerfile hail-ubuntu/Dockerfile.out
	../docker-build.sh hail-ubuntu Dockerfile.out $(HAIL_UBUNTU_IMAGE)
	echo $(HAIL_UBUNTU_IMAGE) > $@

.PHONY: base base-image-ref
base: base-image-ref
base-image-ref: hail-ubuntu-image-ref Dockerfile.base core-site.xml requirements.txt ../pylintrc ../setup.cfg
	python3 ../ci/jinja2_render.py '{"hail_ubuntu_image":{"image":"'$$(cat hail-ubuntu-image-ref)'"}}' Dockerfile.base Dockerfile.base.out
	../docker-build.sh .. docker/Dockerfile.base.out $(BASE_IMAGE)
	echo $(BASE_IMAGE) > $@

.PHONY: hail_version
hail_version:
	$(MAKE) -C ../hail python/hailtop/hail_version
	cp ../hail/python/hailtop/hail_version ../hail_version

.PHONY: service-base service-base-image-ref
service-base: service-base-image-ref
service-base-image-ref: base hail_version
	python3 ../ci/jinja2_render.py '{"base_image":{"image":"'$$(cat base-image-ref)'"}}' Dockerfile.service-base Dockerfile.service-base.out
	../docker-build.sh .. docker/Dockerfile.service-base.out $(SERVICE_BASE_IMAGE)
	echo $(SERVICE_BASE_IMAGE) > $@

.PHONY: service-java-run-base service-java-run-base-image-ref
service-java-run-base: service-java-run-base-image-ref
service-java-run-base-image-ref: hail-ubuntu-image-ref hail_version
	python3 ../ci/jinja2_render.py '{"hail_ubuntu_image":{"image":"'$$(cat hail-ubuntu-image-ref)'"}}' Dockerfile.service-java-run-base Dockerfile.service-java-run-base.out
	../docker-build.sh .. docker/Dockerfile.service-java-run-base.out $(SERVICE_JAVA_RUN_BASE_IMAGE)
	echo $(SERVICE_JAVA_RUN_BASE_IMAGE) > $@

.PHONY: deploy
deploy: build

.PHONY: clean
clean:
	rm -f base hail-ubuntu-image-ref base service-base service-java-run-base
