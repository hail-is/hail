include ../../config.mk

.PHONY: publish-python-dill mirror-dockerhub-images

publish-python-dill:
	bash python-dill/push.sh

mirror-dockerhub-images:
	$(info mirror-dockerhub-images has moved to the target mirror-images in the ../Makefile)
	$(MAKE) -C .. mirror-images
