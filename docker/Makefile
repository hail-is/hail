.PHONY: build push deploy

PROJECT = $(shell gcloud config get-value project)

BASE_IMAGE = gcr.io/$(PROJECT)/base:$(shell docker images -q --no-trunc base:latest | sed -e 's,[^:]*:,,')
SPARK_BASE_IMAGE = gcr.io/$(PROJECT)/spark-base:$(shell docker images -q --no-trunc spark-base:latest | sed -e 's,[^:]*:,,')

build:
	-docker pull ubuntu:18.04
	-docker pull gcr.io/$(PROJECT)/base
	-docker pull gcr.io/$(PROJECT)/spark-base
	docker build . -t base -f Dockerfile.base --cache-from base,ubuntu:18.04
	docker build . -t spark-base -f Dockerfile.spark-base --cache-from spark-base,base,ubuntu:18.04

push: build
	docker tag base $(BASE_IMAGE)
	docker push $(BASE_IMAGE)
	docker tag spark-base $(SPARK_BASE_IMAGE)
	docker push $(SPARK_BASE_IMAGE)

deploy: push
