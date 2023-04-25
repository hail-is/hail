include ../config.mk

.PHONY: build run run-docker deploy clean

WEBSITE_IMAGE := $(DOCKER_PREFIX)/website:$(TOKEN)

build:
	$(MAKE) -C .. website-image

run:
	$(MAKE) -C .. docs
	cd website && tar -xvzf ../../docs.tar.gz
	HAIL_DOMAIN=localhost:5005 python3 -m website local

run-docker: build
	docker run -e HAIL_DOMAIN=localhost:5000 -p 5000:5000 website python3 -m website local

deploy: build
	! [ -z $(NAMESPACE) ]  # call this like: make deploy NAMESPACE=default
	$(MAKE) -C ../docker hail-ubuntu
	python3 ../ci/jinja2_render.py '{"default_ns":{"name":"$(NAMESPACE)"},"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":$(DEPLOY),"scope":"$(SCOPE)","website_image":{"image":"$(WEBSITE_IMAGE)"}}' deployment.yaml deployment.yaml.out
	kubectl -n $(NAMESPACE) apply -f deployment.yaml.out

clean:
	rm -rf deployment.yaml website/docs website/static/css
