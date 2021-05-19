include ../config.mk

.PHONY: build deploy

TOKEN = $(shell cat /dev/urandom | LC_ALL=C tr -dc 'a-z0-9' | head -c 12)

GATEWAY_IMAGE := $(DOCKER_PREFIX)/gateway:$(TOKEN)

build:
	$(MAKE) -C ../docker hail-ubuntu
	python3 ../ci/jinja2_render.py '{"hail_ubuntu_image":{"image":"'$$(cat ../docker/hail-ubuntu-stmp)'"}}' Dockerfile Dockerfile.out
	../docker-build.sh . Dockerfile.out $(GATEWAY_IMAGE)

deploy: build
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":true,"global":{"ip":"$(IP)"}}' service.yaml service.yaml.out
	kubectl -n default apply -f service.yaml.out
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":true,"gateway_image":{"image":"$(GATEWAY_IMAGE)"},"global":{"project":"$(PROJECT)"}}' deployment.yaml deployment.yaml.out
	kubectl -n default apply -f deployment.yaml.out
