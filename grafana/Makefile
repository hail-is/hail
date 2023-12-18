include ../config.mk

.PHONY: build deploy

CLOUD := $(shell kubectl get secret global-config --template={{.data.cloud}} | base64 --decode)

build:
	$(MAKE) -C .. pushed-private-hail-ubuntu-image

deploy: build
	! [ -z $(NAMESPACE) ]  # call this like: make deploy NAMESPACE=default
	python3 ../ci/jinja2_render.py '{"deploy":$(DEPLOY),"global": {"cloud": "$(CLOUD)", "domain": "$(DOMAIN)", "docker_prefix":"$(DOCKER_PREFIX)"},"default_ns":{"name":"$(NAMESPACE)"},"hail_ubuntu_image":{"image":"'$$(cat ../pushed-private-hail-ubuntu-image)'"}}' deployment.yaml deployment.yaml.out
	kubectl -n $(NAMESPACE) apply -f deployment.yaml.out
