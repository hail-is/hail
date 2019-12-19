PROJECT := $(shell gcloud config get-value project)
ZONE := us-central1-a
KUBERNETES_SERVER_URL := https://104.198.230.143

PYTHONPATH := $${PYTHONPATH:+$${PYTHONPATH}:}../hail/python:../gear:../web_common
PYTHON := PYTHONPATH=$(PYTHONPATH) python3

check:
	$(PYTHON) -m flake8  --config ../setup.cfg batch
	$(PYTHON) -m pylint --rcfile ../pylintrc batch --score=n

build:
	make -C ../docker build
	python3 ../ci/jinja2_render.py '{"service_base_image":{"image":"service-base"}}' Dockerfile Dockerfile.out
	docker build -t dbuf -f Dockerfile.out

push: build
	HASH=$(shell docker images -q --no-trunc dbuf | sed -e 's,[^:]*:,,') \
    docker tag dbuf gcr.io/$(PROJECT)/dbuf:$(HASH)
    docker push gcr.io/$(PROJECT)/dbuf:$(HASH)

deploy: push
	HASH=$(shell docker images -q --no-trunc dbuf | sed -e 's,[^:]*:,,') \
    python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":true,"batch_image":{"image":"gcr.io/$(PROJECT)/dbuf:$(HASH)"},"default_ns":{"name":"default"},"global":{"project":"$(PROJECT)","zone":"$(ZONE)","domain":"$(DOMAIN)","k8s_server_url":"$(KUBERNETES_SERVER_URL)"}}' deployment.yaml deployment.yaml.out
	kubectl -n $(NAMESPACE) apply -f deployment.yaml.out
