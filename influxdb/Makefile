include ../config.mk

.PHONY: build push deploy

TOKEN = $(shell cat /dev/urandom | LC_ALL=C tr -dc 'a-z0-9' | head -c 12)

INFLUXDB_NGINX_IMAGE := $(DOCKER_PREFIX)/influxdb_nginx:$(TOKEN)

build:
	$(MAKE) -C ../docker hail-ubuntu
	python3 ../ci/jinja2_render.py '{"hail_ubuntu_image":{"image":"'$$(cat ../docker/hail-ubuntu-image-ref)'"}}' Dockerfile.nginx Dockerfile.nginx.out
	python3 ../ci/jinja2_render.py '{"deploy": $(DEPLOY), "default_ns": {"name": "$(NAMESPACE)"}}' nginx.conf nginx.conf.out
	../docker-build.sh . Dockerfile.nginx.out $(INFLUXDB_NGINX_IMAGE)

deploy: build
	! [ -z $(NAMESPACE) ]  # call this like: make deploy NAMESPACE=default
	python3 ../ci/jinja2_render.py '{"deploy":$(DEPLOY),"default_ns":{"name":"$(NAMESPACE)"}, "influxdb_nginx_image": {"image": "$(INFLUXDB_NGINX_IMAGE)"}}' influx.yaml influx.yaml.out
	kubectl -n $(NAMESPACE) apply -f influx.yaml.out
