include ../config.mk

.PHONY: mirror-images

mirror-images:
	DOCKER_PREFIX=$(DOCKER_PREFIX) \
	HAIL_PIP_VERSION=$(shell cat ../hail/python/hail/hail_pip_version) \
	./mirror_images.sh
