include ../config.mk

.PHONY: build push deploy

PROM_NGINX_LATEST = $(DOCKER_PREFIX)/prom_nginx:latest
PROM_NGINX_IMAGE = $(DOCKER_PREFIX)/prom_nginx:$(shell docker images -q --no-trunc prom_nginx | sed -e 's,[^:]*:,,')

build:
	$(MAKE) -C ../docker hail-ubuntu
	-docker pull $(PROM_NGINX_LATEST)
	python3 ../ci/jinja2_render.py '{"hail_ubuntu_image": {"image": "hail-ubuntu"}}' Dockerfile.nginx Dockerfile.nginx.out
	python3 ../ci/jinja2_render.py '{"deploy": $(DEPLOY), "default_ns": {"name": "$(NAMESPACE)"}}' nginx.conf nginx.conf.out
	docker build -t prom_nginx -f Dockerfile.nginx.out --cache-from prom_nginx,$(PROM_NGINX_LATEST),hail-ubuntu .

push: build
	docker tag prom_nginx $(PROM_NGINX_LATEST)
	docker push $(PROM_NGINX_LATEST)
	docker tag prom_nginx $(PROM_NGINX_IMAGE)
	docker push $(PROM_NGINX_IMAGE)

deploy: push
	! [ -z $(NAMESPACE) ]  # call this like: make deploy NAMESPACE=default
	python3 ../ci/jinja2_render.py '{"deploy":$(DEPLOY),"default_ns":{"name":"$(NAMESPACE)"}, "prom_nginx_image": {"image": "$(PROM_NGINX_IMAGE)"}}' prometheus.yaml prometheus.yaml.out
	kubectl -n $(NAMESPACE) apply -f prometheus.yaml.out
