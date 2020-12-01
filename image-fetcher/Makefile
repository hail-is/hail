include ../config.mk

.PHONY: build push deploy

FETCH_LATEST = gcr.io/$(PROJECT)/image-fetcher:latest
FETCH_IMAGE = gcr.io/$(PROJECT)/image-fetcher:$(shell docker images -q --no-trunc image-fetcher | sed -e 's,[^:]*:,,')

fetch-image.sh: fetch-image.sh.in
	sed -e "s,@project@,$(PROJECT)," \
	  < fetch-image.sh.in > fetch-image.sh

ifeq ($(IN_HAIL_CI),1)
build: fetch-image.sh
	docker pull gcr.io/$(PROJECT)/alpine:3.8
	-docker pull $(FETCH_LATEST)
	python3 ../ci/jinja2_render.py '{"global":{"project":"$(PROJECT)"}}' Dockerfile Dockerfile.out
	docker build -f Dockerfile.out . -t image-fetcher --cache-from image-fetcher,$(FETCH_LATEST),gcr.io/$(PROJECT)/alpine:3.8
else
build: fetch-image.sh
	docker build . -t image-fetcher
endif

push: build
	docker tag image-fetcher $(FETCH_LATEST)
	docker push $(FETCH_LATEST)
	docker tag image-fetcher $(FETCH_IMAGE)
	docker push $(FETCH_IMAGE)

deploy: push
	! [ -z $(NAMESPACE) ]  # call this like: make deploy NAMESPACE=default
	sed -e "s,@sha@,$(shell git rev-parse --short=12 HEAD)," \
	  -e "s,@image@,$(FETCH_IMAGE)," \
	  < deployment.yaml.in > deployment.yaml
	kubectl -n $(NAMESPACE) apply -f deployment.yaml
