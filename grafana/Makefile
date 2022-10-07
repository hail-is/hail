include ../config.mk

.PHONY: build push deploy

TOKEN = $(shell cat /dev/urandom | LC_ALL=C tr -dc 'a-z0-9' | head -c 12)
CLOUD := $(shell kubectl get secret global-config --template={{.data.cloud}} | base64 --decode)

GRAFANA_NGINX_IMAGE := $(DOCKER_PREFIX)/grafana_nginx:$(TOKEN)

build:
	$(MAKE) -C ../docker hail-ubuntu
	$(MAKE) -C ../docker base
	python3 ../ci/jinja2_render.py '{"hail_ubuntu_image":{"image":"'$$(cat ../docker/hail-ubuntu-image-ref)'"}}' Dockerfile.nginx Dockerfile.nginx.out
	python3 ../ci/jinja2_render.py '{"deploy": $(DEPLOY),"global": {"domain": "$(DOMAIN)"},"default_ns": {"name": "$(NAMESPACE)"}}' nginx.conf nginx.conf.out
	../docker-build.sh . Dockerfile.nginx.out $(GRAFANA_NGINX_IMAGE)

deploy: build
	! [ -z $(NAMESPACE) ]  # call this like: make deploy NAMESPACE=default
	python3 ../ci/jinja2_render.py '{"deploy":$(DEPLOY),"global": {"cloud": "$(CLOUD)", "domain": "$(DOMAIN)", "docker_prefix": "$(DOCKER_PREFIX)"},"default_ns":{"name":"$(NAMESPACE)"}, "grafana_nginx_image": {"image": "$(GRAFANA_NGINX_IMAGE)"}, "base_image":{"image":"'$$(cat ../docker/base-image-ref)'"}}' deployment.yaml deployment.yaml.out
	kubectl -n $(NAMESPACE) apply -f deployment.yaml.out
