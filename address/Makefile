PROJECT := $(shell gcloud config get-value project)

ADDRESS_LATEST = gcr.io/$(PROJECT)/address:latest
ADDRESS_IMAGE = gcr.io/$(PROJECT)/address:$(shell docker images -q --no-trunc address | sed -e 's,[^:]*:,,')

PYTHONPATH := $${PYTHONPATH:+$${PYTHONPATH}:}../hail/python:../gear:../web_common
PYTHON := PYTHONPATH=$(PYTHONPATH) python3

.PHONY: check
check:
	$(PYTHON) -m flake8  --config ../setup.cfg address
	$(PYTHON) -m pylint --rcfile ../pylintrc address --score=n

.PHONY: build
build:
	make -C ../docker build
	-docker pull $(ADDRESS_LATEST)
	python3 ../ci/jinja2_render.py '{"service_base_image":{"image":"service-base"}}' Dockerfile Dockerfile.out
	docker build -t address -f Dockerfile.out --cache-from address,$(ADDRESS_LATEST),service-base ..

.PHONY: push
push: build
	docker tag address $(ADDRESS_LATEST)
	docker push $(ADDRESS_LATEST)
	docker tag address $(ADDRESS_IMAGE)
	docker push $(ADDRESS_IMAGE)

.PHONY: deploy
deploy: push
	! [ -z $(NAMESPACE) ]  # call this like: make deploy NAMESPACE=default
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":true,"address_image":{"image":"$(ADDRESS_IMAGE)"},"default_ns":{"name":"$(NAMESPACE)"},"global":{"project":"$(PROJECT)","domain":"$(DOMAIN)"}}' deployment.yaml deployment.yaml.out
	kubectl -n $(NAMESPACE) apply -f service-account.yaml
	kubectl -n $(NAMESPACE) apply -f deployment.yaml.out
