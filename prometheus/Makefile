include ../config.mk

.PHONY: deploy

deploy:
	! [ -z $(NAMESPACE) ]  # call this like: make deploy NAMESPACE=default
	python3 ../ci/jinja2_render.py '{"deploy":$(DEPLOY),"default_ns":{"name":"$(NAMESPACE)"},"global":{"domain":"$(DOMAIN)","docker_prefix":"$(DOCKER_PREFIX)"}}' prometheus.yaml prometheus.yaml.out
	kubectl -n $(NAMESPACE) apply -f prometheus.yaml.out
