include ../config.mk

.PHONY: docs build run run-docker deploy clean

TOKEN = $(shell cat /dev/urandom | LC_ALL=C tr -dc 'a-z0-9' | head -c 12)

WEBSITE_IMAGE := $(DOCKER_PREFIX)/website:$(TOKEN)

check:
	curlylint .

docs:
	$(MAKE) -C ../hail hail-docs-no-test batch-docs
	tar czf docs.tar.gz -C ../hail/build/www .

build: docs
	$(MAKE) -C ../docker service-base
	python3 ../ci/jinja2_render.py '{"service_base_image": {"image":"'$$(cat ../docker/service-base-image-ref)'"}}' Dockerfile Dockerfile.out
	../docker-build.sh . Dockerfile.out $(WEBSITE_IMAGE)

run: docs
	cd website && tar -xvzf ../docs.tar.gz
	HAIL_DOMAIN=localhost:5000 python3 -m website local

run-docker: build
	docker run -e HAIL_DOMAIN=localhost:5000 -p 5000:5000 website python3 -m website local

deploy: build
	! [ -z $(NAMESPACE) ]  # call this like: make deploy NAMESPACE=default
	$(MAKE) -C ../docker hail-ubuntu
	python3 ../ci/jinja2_render.py '{"global": {"domain":"$(DOMAIN)"},"default_ns":{"name":"$(NAMESPACE)"},"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":$(DEPLOY),"website_image":{"image":"$(WEBSITE_IMAGE)"}}' deployment.yaml deployment.yaml.out
	kubectl -n $(NAMESPACE) apply -f deployment.yaml.out

clean:
	rm -rf deployment.yaml
	rm -rf website/docs
	rm -rf website/static/css
