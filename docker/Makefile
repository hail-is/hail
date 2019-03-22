.PHONY: build clean

build: build-stmp

ifeq ($(IN_HAIL_CI),1)
.PHONY: push deploy

PROJECT = $(shell gcloud config get-value project)
BASE_LATEST = gcr.io/$(PROJECT)/base:latest
BASE_IMAGE = gcr.io/$(PROJECT)/base:$(shell docker images -q --no-trunc base:latest | sed -e 's,[^:]*:,,')

build-stmp: Dockerfile.base core-site.xml requirements.txt
	-docker pull ubuntu:18.04
	-docker pull $(BASE_LATEST)
	docker build -t base -f Dockerfile.base --cache-from base:latest,ubuntu:18.04 .
	touch build-stmp

push: build
	docker tag base $(BASE_LATEST)
	docker push $(BASE_LATEST)
	docker tag base $(BASE_IMAGE)
	docker push $(BASE_IMAGE)

deploy: push
else
build-stmp: Dockerfile.base core-site.xml requirements.txt
	docker build -t base -f Dockerfile.base .
	touch build-stmp
endif

clean:
	rm -f build-stmp
