include ../config.mk

.PHONY: build push deploy

TOKEN = $(shell cat /dev/urandom | LC_ALL=C tr -dc 'a-z0-9' | head -c 12)

PROM_NGINX_IMAGE := $(DOCKER_PREFIX)/prom_nginx:$(TOKEN)

build:
	$(MAKE) -C ../docker hail-ubuntu
	python3 ../ci/jinja2_render.py '{"hail_ubuntu_image": {"image": "'$$(cat ../docker/hail-ubuntu-image-ref)'"}}' Dockerfile.nginx Dockerfile.nginx.out
	python3 ../ci/jinja2_render.py '{"deploy": $(DEPLOY), "default_ns": {"name": "$(NAMESPACE)"},"global":{"domain":"$(DOMAIN)"}}' nginx.conf nginx.conf.out
	../docker-build.sh . Dockerfile.nginx.out $(PROM_NGINX_IMAGE)

deploy: build
	! [ -z $(NAMESPACE) ]  # call this like: make deploy NAMESPACE=default
	python3 ../ci/jinja2_render.py '{"deploy":$(DEPLOY),"default_ns":{"name":"$(NAMESPACE)"},"global":{"domain":"$(DOMAIN)"},"prom_nginx_image": {"image": "$(PROM_NGINX_IMAGE)"}}' prometheus.yaml prometheus.yaml.out
	kubectl -n $(NAMESPACE) apply -f prometheus.yaml.out
