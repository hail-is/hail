include ../config.mk

NOTEBOOK_LATEST = $(DOCKER_PREFIX)/notebook:latest
NOTEBOOK_IMAGE = $(DOCKER_PREFIX)/notebook:$(shell docker images -q --no-trunc notebook | sed -e 's,[^:]*:,,')
NOTEBOOK_NGINX_LATEST = $(DOCKER_PREFIX)/notebook_nginx:latest
NOTEBOOK_NGINX_IMAGE = $(DOCKER_PREFIX)/notebook_nginx:$(shell docker images -q --no-trunc notebook_nginx | sed -e 's,[^:]*:,,')

EXTRA_PYTHONPATH := ../hail/python:../gear:../web_common
PYTHON := PYTHONPATH=$${PYTHONPATH:+$${PYTHONPATH}:}$(EXTRA_PYTHONPATH) python3

.PHONY: check
check:
	$(PYTHON) -m flake8 notebook
	$(PYTHON) -m pylint --rcfile ../pylintrc notebook --score=n
	curlylint .
	bash ../check-sql.sh

.PHONY: build-notebook
build-notebook:
	$(MAKE) -C ../docker build
	-docker pull $(NOTEBOOK_LATEST)
	python3 ../ci/jinja2_render.py '{"service_base_image":{"image":"service-base"}}' Dockerfile Dockerfile.out
	docker build -t notebook -f Dockerfile.out --cache-from notebook,$(NOTEBOOK_LATEST),service-base ..

.PHONY: build-nginx
build-nginx:
	$(MAKE) -C ../docker hail-ubuntu
	-docker pull $(NOTEBOOK_NGINX_LATEST)
	python3 ../ci/jinja2_render.py '{"hail_ubuntu_image":{"image":"hail-ubuntu"}}' Dockerfile.nginx Dockerfile.nginx.out
	python3 ../ci/jinja2_render.py '{"deploy": $(DEPLOY), "default_ns": {"name": "$(NAMESPACE)"}}' nginx.conf nginx.conf.out
	docker build -t notebook_nginx -f Dockerfile.nginx.out --cache-from notebook_nginx,$(NOTEBOOK_NGINX_LATEST),hail-ubuntu .

.PHONY: push
push: build-notebook build-nginx
	docker tag notebook $(NOTEBOOK_LATEST)
	docker push $(NOTEBOOK_LATEST)
	docker tag notebook $(NOTEBOOK_IMAGE)
	docker push $(NOTEBOOK_IMAGE)
	docker tag notebook_nginx $(NOTEBOOK_NGINX_LATEST)
	docker push $(NOTEBOOK_NGINX_LATEST)
	docker tag notebook_nginx $(NOTEBOOK_NGINX_IMAGE)
	docker push $(NOTEBOOK_NGINX_IMAGE)


JINJA_ENVIRONMENT = '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":$(DEPLOY),"notebook_image":{"image":"$(NOTEBOOK_IMAGE)"},"notebook_nginx_image":{"image":"$(NOTEBOOK_NGINX_IMAGE)"},"default_ns":{"name":"$(NAMESPACE)"},"notebook_database":{"user_secret_name":"sql-notebook-user-config"},"global":{"project":"$(PROJECT)","domain":"$(DOMAIN)","k8s_server_url":"$(KUBERNETES_SERVER_URL)"},"scope":"$(SCOPE)"}'

.PHONY: deploy
deploy: push
	! [ -z $(NAMESPACE) ]  # call this like: make deploy NAMESPACE=default
	E=$(JINJA_ENVIRONMENT) && \
	  python3 ../ci/jinja2_render.py $$E deployment.yaml deployment.yaml.out && \
	kubectl -n $(NAMESPACE) apply -f deployment.yaml.out
