include ../config.mk

.PHONY: build push deploy

TOKEN = $(shell cat /dev/urandom | LC_ALL=C tr -dc 'a-z0-9' | head -c 12)

GATEWAY_IMAGE := $(DOCKER_PREFIX)/gateway:$(TOKEN)
IP := $(shell kubectl get secret global-config --template={{.data.ip}} | base64 --decode)

export NAMESPACE := default

build:
	$(MAKE) -C ../docker hail-ubuntu
	python3 ../ci/jinja2_render.py '{"hail_ubuntu_image":{"image":"'$$(cat ../docker/hail-ubuntu-image-ref)'"}}' Dockerfile Dockerfile.out
	python3 ../ci/jinja2_render.py '{"domain": "$(DOMAIN)","subdomains":[$(shell sed -e 's/\(.*\)/"\1"/' ../letsencrypt/subdomains.txt | paste -s -d, -)]}' gateway.nginx.conf gateway.nginx.conf.out
	../docker-build.sh . Dockerfile.out $(GATEWAY_IMAGE)

deploy: build
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"global":{"ip":"$(IP)"}}' service.yaml service.yaml.out
	kubectl -n default apply -f service.yaml.out
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"gateway_image":{"image":"$(GATEWAY_IMAGE)"}}' deployment.yaml deployment.yaml.out
	kubectl -n default apply -f deployment.yaml.out
