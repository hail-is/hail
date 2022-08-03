include ../config.mk

.PHONY: build push deploy

TOKEN = $(shell cat /dev/urandom | LC_ALL=C tr -dc 'a-z0-9' | head -c 12)

INTERNAL_GATEWAY_IMAGE := $(DOCKER_PREFIX)/internal-gateway:$(TOKEN)
INTERNAL_IP := $(shell kubectl get secret global-config --template={{.data.internal_ip}} | base64 --decode)

export NAMESPACE := default

build:
	# python3 ../ci/jinja2_render.py '{"hail_ubuntu_image":{"image":"'$$(cat ../docker/hail-ubuntu-image-ref)'"}}' Dockerfile Dockerfile.out
	../docker-build.sh . Dockerfile $(INTERNAL_GATEWAY_IMAGE)

deploy: build
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"global":{"internal_ip":"$(INTERNAL_IP)"}}' service.yaml service.yaml.out
	kubectl -n default apply -f service.yaml.out
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"internal_gateway_image":{"image":"$(INTERNAL_GATEWAY_IMAGE)"}}' deployment.yaml deployment.yaml.out
	kubectl -n default apply -f deployment.yaml.out
