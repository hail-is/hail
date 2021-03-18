include ../config.mk

.PHONY: build push deploy

GRAFANA_NGINX_LATEST = gcr.io/$(PROJECT)/grafana_nginx:latest
GRAFANA_NGINX_IMAGE = gcr.io/$(PROJECT)/grafana_nginx:$(shell docker images -q --no-trunc grafana_nginx | sed -e 's,[^:]*:,,')

build:
	$(MAKE) -C ../docker hail-ubuntu
	-docker pull $(GRAFANA_NGINX_LATEST)
	python3 ../ci/jinja2_render.py '{"hail_ubuntu_image":{"image":"hail-ubuntu"}}' Dockerfile.nginx Dockerfile.nginx.out
	python3 ../ci/jinja2_render.py '{"deploy": $(DEPLOY), "default_ns": {"name": "$(NAMESPACE)"}}' nginx.conf nginx.conf.out
	docker build -t grafana_nginx -f Dockerfile.nginx.out --cache-from grafana_nginx,$(GRAFANA_NGINX_LATEST),hail-ubuntu .

push: build
	docker tag grafana_nginx $(GRAFANA_NGINX_LATEST)
	docker push $(GRAFANA_NGINX_LATEST)
	docker tag grafana_nginx $(GRAFANA_NGINX_IMAGE)
	docker push $(GRAFANA_NGINX_IMAGE)

deploy: push
	! [ -z $(NAMESPACE) ]  # call this like: make deploy NAMESPACE=default
	python3 ../ci/jinja2_render.py '{"deploy":$(DEPLOY),"default_ns":{"name":"$(NAMESPACE)"}, "grafana_nginx_image": {"image": "$(GRAFANA_NGINX_IMAGE)"}}' deployment.yaml deployment.yaml.out
	kubectl -n $(NAMESPACE) apply -f deployment.yaml.out
