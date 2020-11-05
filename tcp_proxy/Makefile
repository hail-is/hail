include ../config.mk

TCP_PROXY_LATEST = gcr.io/$(PROJECT)/tcp-proxy:latest
TCP_PROXY_IMAGE = gcr.io/$(PROJECT)/tcp-proxy:$(shell docker images -q --no-trunc tcp-proxy | sed -e 's,[^:]*:,,')

EXTRA_PYTHONPATH := ../hail/python:../gear:../web_common
PYTHON := PYTHONPATH=$${PYTHONPATH:+$${PYTHONPATH}:}$(EXTRA_PYTHONPATH) python3

.PHONY: check
check:
	$(PYTHON) -m flake8  --config ../setup.cfg tcp_proxy
	$(PYTHON) -m pylint --rcfile ../pylintrc tcp_proxy --score=n

.PHONY: build
build:
	$(MAKE) -C ../docker build
	-docker pull $(TCP_PROXY_LATEST)
	python3 ../ci/jinja2_render.py '{"service_base_image":{"image":"service-base"}}' Dockerfile Dockerfile.out
	docker build -t tcp-proxy -f Dockerfile.out --cache-from tcp-proxy,$(TCP_PROXY_LATEST),service-base .

.PHONY: push
push: build
	docker tag tcp-proxy $(TCP_PROXY_LATEST)
	docker push $(TCP_PROXY_LATEST)
	docker tag tcp-proxy $(TCP_PROXY_IMAGE)
	docker push $(TCP_PROXY_IMAGE)

JINJA_ENVIRONMENT = '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":true,"tcp_proxy_image":{"image":"$(TCP_PROXY_IMAGE)"},"default_ns":{"name":"$(NAMESPACE)"},"global":{"project":"$(PROJECT)","domain":"$(DOMAIN)","k8s_server_url":"$(KUBERNETES_SERVER_URL)"},"scope":"deploy"}'

.PHONY: deploy
deploy: push
	! [ -z $(NAMESPACE) ]  # set NAMESPACE
	python3 ../ci/jinja2_render.py $(JINJA_ENVIRONMENT) deployment.yaml deployment.yaml.out
	kubectl -n $(NAMESPACE) apply -f deployment.yaml.out
