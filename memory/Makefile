PROJECT := $(shell gcloud config get-value project)

PYTHONPATH := $${PYTHONPATH:+$${PYTHONPATH}:}../hail/python:../gear
PYTHON := PYTHONPATH=$(PYTHONPATH) python3

MEMORY_LATEST = gcr.io/$(PROJECT)/memory:latest
MEMORY_IMAGE = gcr.io/$(PROJECT)/memory:$(shell docker images -q --no-trunc memory | sed -e 's,[^:]*:,,')

.PHONY: check
check:
	$(PYTHON) -m flake8  --config ../setup.cfg memory
	$(PYTHON) -m pylint --rcfile ../pylintrc memory --score=n

.PHONY: build
build:
	make -C ../docker build
	-docker pull $(MEMORY_LATEST)
	python3 ../ci/jinja2_render.py '{"service_base_image":{"image":"service-base"}}' Dockerfile Dockerfile.out
	docker build -t memory -f Dockerfile.out --cache-from memory,$(MEMORY_LATEST),service-base ..

.PHONY: push
push: build
	docker tag memory $(MEMORY_LATEST)
	docker push $(MEMORY_LATEST)
	docker tag memory $(MEMORY_IMAGE)
	docker push $(MEMORY_IMAGE)

.PHONY: deploy
deploy: push
	kubectl -n default apply -f service-account.yaml
	python3 ../ci/jinja2_render.py '{"default_ns":{"name":"default"}}' service-account-batch-pods.yaml service-account-batch-pods.yaml.out
	kubectl -n batch-pods apply -f service-account-batch-pods.yaml.out
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":true,"memory_image":{"image":"$(MEMORY_IMAGE)"},"batch_pods_ns":{"name":"batch-pods"},"global":{"project":"$(PROJECT)","domain":"$(DOMAIN)"}}' deployment.yaml deployment.yaml.out
	kubectl -n default apply -f deployment.yaml.out
