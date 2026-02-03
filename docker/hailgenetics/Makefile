include ../../config.mk

.PHONY: publish-python-dill mirror-dockerhub-images

publish-python-dill:
	bash python-dill/push.sh

mirror-dockerhub-images:
	DOCKER_PREFIX=$(DOCKER_PREFIX) \
	DOCKERHUB_PREFIX=$(DOCKERHUB_PREFIX) \
	HAIL_PIP_VERSION=$(shell cat ../../hail/env/HAIL_PIP_VERSION) \
	./mirror_images.sh
