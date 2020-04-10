.PHONY: build
build: base-stmp service-base

PROJECT := $(shell gcloud config get-value project)

BASE_LATEST = gcr.io/$(PROJECT)/base:latest
BASE_IMAGE = gcr.io/$(PROJECT)/base:$(shell docker images -q --no-trunc base:latest | sed -e 's,[^:]*:,,')

SERVICE_BASE_LATEST = gcr.io/$(PROJECT)/service-base:latest
SERVICE_BASE_IMAGE = gcr.io/$(PROJECT)/service-base:$(shell docker images -q --no-trunc service-base:latest | sed -e 's,[^:]*:,,')

base-stmp: Dockerfile.base core-site.xml requirements.txt ../pylintrc ../setup.cfg
	-docker pull ubuntu:18.04
	-docker pull $(BASE_LATEST)
	docker build -t base -f Dockerfile.base --cache-from base,$(BASE_LATEST),ubuntu:18.04 ..
	touch base-stmp

.PHONY: service-base
service-base:
	-docker pull $(SERVICE_BASE_LATEST)
	python3 ../ci/jinja2_render.py '{"base_image":{"image":"base"}}' Dockerfile.service-base Dockerfile.service-base.out
	docker build -t service-base -f Dockerfile.service-base.out --cache-from service-base,$(SERVICE_BASE_LATEST),base ..

.PHONY: hail-public-image
hail-public-image:
	make -C ../hail wheel
	cp ../hail/build/deploy/dist/hail-*-py3-none-any.whl hail/
	cd hail && tar -cvf wheel-container.tar hail-*-py3-none-any.whl
	docker build hail -f hail/Dockerfile -t hail-public-image

.PHONY: genetics-public-image
genetics-public-image: hail-public-image
	python3 ../ci/jinja2_render.py '{"hail_public_image":{"image":"hail-public-image"}}' genetics/Dockerfile genetics/Dockerfile.out
	docker build genetics -f genetics/Dockerfile.out -t genetics-public-image

.PHONY: push
push: build
	docker tag base $(BASE_LATEST)
	docker push $(BASE_LATEST)
	docker tag base $(BASE_IMAGE)
	docker push $(BASE_IMAGE)
	docker tag service-base $(SERVICE_BASE_LATEST)
	docker push $(SERVICE_BASE_LATEST)
	docker tag service-base $(SERVICE_BASE_IMAGE)
	docker push $(SERVICE_BASE_IMAGE)

.PHONY: deploy
deploy: push

.PHONY: clean
clean:
	rm -f build-stmp
