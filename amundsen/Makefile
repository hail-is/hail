include ../config.mk

PYTHON := PYTHONPATH=$${PYTHONPATH:+$${PYTHONPATH}:}$(EXTRA_PYTHONPATH) python3

JINJA_ENVIRONMENT = '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":$(DEPLOY),"default_ns":{"name":"$(NAMESPACE)"},"batch_pods_ns":{"name":"$(BATCH_PODS_NAMESPACE)"},"global":{"project":"$(PROJECT)","domain":"$(DOMAIN)","k8s_server_url":"$(KUBERNETES_SERVER_URL)"},"scope":"$(SCOPE)"}'

.PHONY: deploy
deploy:
	! [ -z $(NAMESPACE) ]  # call this like: make deploy NAMESPACE=default
	E=$(JINJA_ENVIRONMENT) && \
	  python3 ../ci/jinja2_render.py $$E deployment.yaml deployment.yaml.out
	kubectl -n $(NAMESPACE) apply -f deployment.yaml.out
