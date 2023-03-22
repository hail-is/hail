include ../config.mk

.PHONY: docs build run run-docker deploy clean

WEBSITE_IMAGE := $(DOCKER_PREFIX)/website:$(TOKEN)

EXTRA_PYTHONPATH := ../hail/python:../gear:../web_common
PYTHON := PYTHONPATH=$${PYTHONPATH:+$${PYTHONPATH}:}$(EXTRA_PYTHONPATH) python3

BLACK := $(PYTHON) -m black . --line-length=120 --skip-string-normalization

check:
	$(PYTHON) -m isort . --check-only --diff
	$(BLACK) --check --diff
	curlylint .

docs:
	$(MAKE) -C ../hail hail-docs-no-test batch-docs
	gcloud storage cp gs://hail-common/builds/0.1/docs/hail-0.1-docs-5a6778710097.tar.gz .
	mkdir -p ../hail/build/www/docs/0.1
	tar -xvf hail-0.1-docs-5a6778710097.tar.gz -C ../hail/build/www/docs/0.1 --strip-components 2
	rm hail-0.1-docs-5a6778710097.tar.gz
	tar czf docs.tar.gz -C ../hail/build/www .

build: docs
	$(MAKE) -C ../docker hail-ubuntu
	python3 ../ci/jinja2_render.py '{"hail_ubuntu_image": {"image":"'$$(cat ../docker/hail-ubuntu-image-ref)'"}}' Dockerfile Dockerfile.out
	../docker-build.sh . Dockerfile.out $(WEBSITE_IMAGE)

run: docs
	cd website && tar -xvzf ../docs.tar.gz
	HAIL_DOMAIN=localhost:5000 python3 -m website local

run-docker: build
	docker run -e HAIL_DOMAIN=localhost:5000 -p 5000:5000 website python3 -m website local

deploy: build
	! [ -z $(NAMESPACE) ]  # call this like: make deploy NAMESPACE=default
	$(MAKE) -C ../docker hail-ubuntu
	python3 ../ci/jinja2_render.py '{"default_ns":{"name":"$(NAMESPACE)"},"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":$(DEPLOY),"scope":"$(SCOPE)","website_image":{"image":"$(WEBSITE_IMAGE)"}}' deployment.yaml deployment.yaml.out
	kubectl -n $(NAMESPACE) apply -f deployment.yaml.out

clean:
	rm -rf deployment.yaml
	rm -rf website/docs
	rm -rf website/static/css
