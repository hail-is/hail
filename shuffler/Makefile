include ../config.mk

SHUFFLER_LATEST = $(DOCKER_PREFIX)/shuffler:latest
SHUFFLER_IMAGE = $(DOCKER_PREFIX)/shuffler:$(shell docker images -q --no-trunc shuffler | sed -e 's,[^:]*:,,')

PYTHONPATH := $${PYTHONPATH:+$${PYTHONPATH}:}../hail/python:../gear:../web_common
PYTHON := PYTHONPATH=$(PYTHONPATH) python3

.PHONY: build
build:
	$(MAKE) -C ../docker build
	-docker pull $(SHUFFLER_LATEST)
	python3 ../ci/jinja2_render.py '{"service_base_image":{"image":"service-base"}}' Dockerfile Dockerfile.out
	$(MAKE) -C ../hail shadowJar
	cp ../hail/build/libs/hail-all-spark.jar hail.jar
	docker build -t shuffler -f Dockerfile.out --cache-from shuffler,$(SHUFFLER_LATEST),service-base .

.PHONY: push
push: build
	docker tag shuffler $(SHUFFLER_LATEST)
	docker push $(SHUFFLER_LATEST)
	docker tag shuffler $(SHUFFLER_IMAGE)
	docker push $(SHUFFLER_IMAGE)

.PHONY: deploy
deploy: push
	! [ -z $(NAMESPACE) ]  # call this like: make deploy NAMESPACE=default
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":$(DEPLOY),"shuffler_image":{"image":"$(SHUFFLER_IMAGE)"},"default_ns":{"name":"$(NAMESPACE)"},"global":{"project":"$(PROJECT)","domain":"$(DOMAIN)","k8s_server_url":"$(KUBERNETES_SERVER_URL)"}}' deployment.yaml deployment.yaml.out
	kubectl -n $(NAMESPACE) apply -f deployment.yaml.out
