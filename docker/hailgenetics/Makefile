include ../../config.mk

.PHONY: publish-python-dill mirror-dockerhub-images

publish-python-dill:
	DOCKER_PREFIX=$(DOCKER_PREFIX) bash python-dill/push.sh

mirror-dockerhub-images:
	DOCKER_PREFIX=$(DOCKER_PREFIX) \
	HAIL_PIP_VERSION=$(shell cat ../../hail/python/hail/hail_pip_version) \
	./mirror_images.sh
