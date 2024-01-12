include ../config.mk

.PHONY: deploy
deploy:
	! [ -z $(NAMESPACE) ]  # call this like: make deploy NAMESPACE=default
	python3 ../ci/jinja2_render.py \
		'{"global":{"docker_prefix":"$(DOCKER_PREFIX)"},"default_ns":{"name":"$(NAMESPACE)"}}' \
		deployment.yaml deployment.yaml.out
	kubectl -n $(NAMESPACE) apply -f deployment.yaml.out
