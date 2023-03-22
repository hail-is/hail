include ../config.mk

.PHONY: build deploy

CLOUD := $(shell kubectl get secret global-config --template={{.data.cloud}} | base64 --decode)

build:
	$(MAKE) -C ../docker base

deploy: build
	! [ -z $(NAMESPACE) ]  # call this like: make deploy NAMESPACE=default
	python3 ../ci/jinja2_render.py '{"deploy":$(DEPLOY),"global": {"cloud": "$(CLOUD)", "domain": "$(DOMAIN)", "docker_prefix":"$(DOCKER_PREFIX)"},"default_ns":{"name":"$(NAMESPACE)"},"base_image":{"image":"'$$(cat ../docker/base-image-ref)'"}}' deployment.yaml deployment.yaml.out
	kubectl -n $(NAMESPACE) apply -f deployment.yaml.out
