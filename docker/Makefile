.PHONY:
build: build-stmp

PROJECT := $(shell gcloud config get-value project)

BASE_LATEST = gcr.io/$(PROJECT)/base:latest
BASE_IMAGE = gcr.io/$(PROJECT)/base:$(shell docker images -q --no-trunc base:latest | sed -e 's,[^:]*:,,')

SERVICE_BASE_LATEST = gcr.io/$(PROJECT)/service-base:latest
SERVICE_BASE_IMAGE = gcr.io/$(PROJECT)/service-base:$(shell docker images -q --no-trunc service-base:latest | sed -e 's,[^:]*:,,')

build-stmp: Dockerfile.base core-site.xml requirements.txt
	-docker pull ubuntu:18.04
	-docker pull $(BASE_LATEST)
	docker build -t base -f Dockerfile.base --cache-from base,$(BASE_LATEST),ubuntu:18.04 ..
	-docker pull $(SERVICE_BASE_LATEST)
	python3 ../ci/jinja2_render.py '{"base_image":{"image":"base"}}' Dockerfile.service-base Dockerfile.service-base.out
	docker build -t service-base -f Dockerfile.service-base.out --cache-from service-base,$(SERVICE_BASE_LATEST),base ..
	touch build-stmp

.PHONY: push
push: build
	docker tag base $(BASE_LATEST)
	docker push $(BASE_LATEST)
	docker tag base $(BASE_IMAGE)
	docker push $(BASE_IMAGE)
	docker tag base $(SERVICE_BASE_LATEST)
	docker push $(SERVICE_BASE_LATEST)
	docker tag base $(SERVICE_BASE_IMAGE)
	docker push $(SERVICE_BASE_IMAGE)

.PHONY: deploy
deploy: push

.PHONY: clean
clean:
	rm -f build-stmp
