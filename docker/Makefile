.PHONY: build clean

build: build-stmp

build-stmp: Dockerfile.base
ifeq ($(IN_HAIL_CI),1)
	-docker pull ubuntu:18.04
	-docker pull gcr.io/$(PROJECT)/base
	docker build -t base -f Dockerfile.base --cache-from base,ubuntu:18.04 .
else
	docker build -t base -f Dockerfile.base .
endif
	touch build-stmp

ifeq ($(IN_HAIL_CI),1)
.PHONY: push deploy

PROJECT = $(shell gcloud config get-value project)
BASE_IMAGE = gcr.io/$(PROJECT)/base:$(shell docker images -q --no-trunc base:latest | sed -e 's,[^:]*:,,')

push: build
	docker push base
	docker tag base $(BASE_IMAGE)
	docker push $(BASE_IMAGE)

deploy: push
endif

clean:
	rm -f build-stmp
