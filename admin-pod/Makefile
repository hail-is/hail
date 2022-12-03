include ../config.mk

ADMIN_POD_IMAGE := $(DOCKER_PREFIX)/admin-pod:$(TOKEN)

.PHONY: hail-ubuntu
hail-ubuntu:
	$(MAKE) -C ../docker hail-ubuntu

.PHONY: build
build: hail-ubuntu
	python3 ../ci/jinja2_render.py '{"hail_ubuntu_image":{"image":"'$$(cat ../docker/hail-ubuntu-image-ref)'"}}' Dockerfile Dockerfile.out
	../docker-build.sh . Dockerfile.out $(ADMIN_POD_IMAGE)

.PHONY: deploy
deploy:
	! [ -z $(NAMESPACE) ]  # call this like: make deploy NAMESPACE=default
	$(MAKE) -C ../docker service-base
	python3 ../ci/jinja2_render.py '{"deploy":$(DEPLOY),"admin_pod_image":{"image":"$(ADMIN_POD_IMAGE)"}}' admin-pod.yaml admin-pod.yaml.out
	kubectl -n $(NAMESPACE) apply -f admin-pod.yaml.out
