include ../config.mk

.PHONY: build deploy

GATEWAY_IMAGE := $(DOCKER_PREFIX)/gateway:$(TOKEN)
IP := $(shell kubectl get secret global-config --template={{.data.ip}} | base64 --decode)

build:
	$(MAKE) -C .. hail-ubuntu-image
	python3 ../ci/jinja2_render.py '{"hail_ubuntu_image":{"image":"'$$(cat ../hail-ubuntu-image)'"}}' Dockerfile Dockerfile.out
	../docker-build.sh . Dockerfile.out $(GATEWAY_IMAGE)

deploy: build
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":true,"global":{"ip":"$(IP)"}}' service.yaml service.yaml.out
	kubectl -n default apply -f service.yaml.out
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"gateway_image":{"image":"$(GATEWAY_IMAGE)"}}' deployment.yaml deployment.yaml.out
	kubectl -n default apply -f deployment.yaml.out
