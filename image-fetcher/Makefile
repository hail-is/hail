include ../config.mk

.PHONY: build push deploy

FETCH_LATEST = $(DOCKER_PREFIX)/image-fetcher:latest
FETCH_IMAGE = $(DOCKER_PREFIX)/image-fetcher:$(shell docker images -q --no-trunc image-fetcher | sed -e 's,[^:]*:,,')

build: fetch-image.sh
	docker pull $(DOCKER_PREFIX)/alpine:3.8
	-docker pull $(FETCH_LATEST)
	python3 ../ci/jinja2_render.py '{"global":{"docker_prefix":"$(DOCKER_PREFIX)"}}' Dockerfile Dockerfile.out
	docker build -f Dockerfile.out . -t image-fetcher --cache-from image-fetcher,$(FETCH_LATEST),$(DOCKER_PREFIX)/alpine:3.8

push: build
	docker tag image-fetcher $(FETCH_LATEST)
	docker push $(FETCH_LATEST)
	docker tag image-fetcher $(FETCH_IMAGE)
	docker push $(FETCH_IMAGE)

deploy: push
	! [ -z $(NAMESPACE) ]  # call this like: make deploy NAMESPACE=default
	python3 ../ci/jinja2_render.py '{"deploy": $(DEPLOY), "code":{"sha": "$(shell git rev-parse --short=12 HEAD)"}, "image_fetcher_image":{"image":"$(FETCH_IMAGE)"}}' deployment.yaml deployment.yaml.out
	kubectl -n $(NAMESPACE) apply -f deployment.yaml.out
