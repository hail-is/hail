include ../config.mk

NOTEBOOK_NGINX_IMAGE := $(DOCKER_PREFIX)/notebook_nginx:$(TOKEN)

.PHONY: build-notebook
build-notebook:
	$(MAKE) -C .. notebook-image

.PHONY: build-nginx
build-nginx:
	$(MAKE) -C .. hail-ubuntu-image
	python3 ../ci/jinja2_render.py '{"hail_ubuntu_image":{"image":"'$$(cat ../hail-ubuntu-image)'"}}' Dockerfile.nginx Dockerfile.nginx.out
	python3 ../ci/jinja2_render.py '{"deploy": $(DEPLOY), "default_ns": {"name": "$(NAMESPACE)"}}' nginx.conf nginx.conf.out
	../docker-build.sh . Dockerfile.nginx.out $(NOTEBOOK_NGINX_IMAGE)

.PHONY: build
build: build-notebook build-nginx

JINJA_ENVIRONMENT = '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":$(DEPLOY),"notebook_image":{"image":"$(shell cat ../notebook-image)"},"notebook_nginx_image":{"image":"$(NOTEBOOK_NGINX_IMAGE)"},"default_ns":{"name":"$(NAMESPACE)"},"notebook_database":{"user_secret_name":"sql-notebook-user-config"},"scope":"$(SCOPE)"}'

.PHONY: deploy
deploy: build
	! [ -z $(NAMESPACE) ]  # call this like: make deploy NAMESPACE=default
	E=$(JINJA_ENVIRONMENT) && \
	  python3 ../ci/jinja2_render.py $$E deployment.yaml deployment.yaml.out && \
	kubectl -n $(NAMESPACE) apply -f deployment.yaml.out
