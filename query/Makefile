include ../config.mk

EXTRA_PYTHONPATH := ../hail/python:../gear
PYTHON := PYTHONPATH=$${PYTHONPATH:+$${PYTHONPATH}:}$(EXTRA_PYTHONPATH) python3

QUERY_LATEST = $(DOCKER_PREFIX)/query:latest
QUERY_IMAGE = $(DOCKER_PREFIX)/query:$(shell docker images -q --no-trunc query | sed -e 's,[^:]*:,,')

.PHONY: check
check:
	$(PYTHON) -m flake8  --config ../setup.cfg query
	$(PYTHON) -m pylint --rcfile ../pylintrc query --score=n

.PHONY: build
build:
	$(MAKE) -C ../docker service-java-run-base
	$(MAKE) -C ../hail shadowJar
# janky
	cp ../hail/build/libs/hail-all-spark.jar ./hail.jar
	-docker pull $(QUERY_LATEST)
	python3 ../ci/jinja2_render.py '{"service_java_run_base_image":{"image":"service-java-run-base"}}' Dockerfile Dockerfile.out
	docker build -t query -f Dockerfile.out --cache-from query,$(QUERY_LATEST),service-base ..

.PHONY: push
push: build
	docker tag query $(QUERY_LATEST)
	docker push $(QUERY_LATEST)
	docker tag query $(QUERY_IMAGE)
	docker push $(QUERY_IMAGE)

.PHONY: deploy
deploy: push
	! [ -z $(NAMESPACE) ]  # call this like: make deploy NAMESPACE=default
	kubectl -n $(NAMESPACE) apply -f service-account.yaml
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":$(DEPLOY),"default_ns":{"name":"$(NAMESPACE)"},"query_image":{"image":"$(QUERY_IMAGE)"},"global":{"project":"$(PROJECT)","domain":"$(DOMAIN)"}}' deployment.yaml deployment.yaml.out
	kubectl -n $(NAMESPACE) apply -f deployment.yaml.out
