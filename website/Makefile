include ../config.mk

.PHONY: build push run run-docker deploy clean

build:
	$(MAKE) -C .. website-image

run:
	$(MAKE) -C .. website/website/docs
	HAIL_DOMAIN=localhost:5555 PORT=5555 python3 -m website local

run-docker: build
	docker run -e HAIL_DOMAIN=localhost:5555 -e PORT=5555 -p 5555:5555 $(shell cat ../website-image) python3 -m website local

push:
	$(MAKE) -C .. pushed-private-website-image

deploy: push
	! [ -z $(NAMESPACE) ]  # call this like: make deploy NAMESPACE=default
	python3 ../ci/jinja2_render.py '{"default_ns":{"name":"$(NAMESPACE)"},"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":$(DEPLOY),"scope":"$(SCOPE)","website_image":{"image":"'$$(cat ../pushed-private-website-image)'"}}' deployment.yaml deployment.yaml.out
	kubectl -n $(NAMESPACE) apply -f deployment.yaml.out

clean:
	rm -rf deployment.yaml website/docs website/static/css
