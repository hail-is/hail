.PHONY: build

ifeq ($(IN_HAIL_CI),1)
.PHONY: push deploy

PROJECT := $(shell gcloud config get-value project)
CI2_LATEST = gcr.io/$(PROJECT)/ci2:latest
CI2_IMAGE = gcr.io/$(PROJECT)/ci2:$(shell docker images -q --no-trunc ci2 | sed -e 's,[^:]*:,,')

build:
	make -C ../docker build
	-docker	pull $(CI2_LATEST)
	docker build -t ci2 -f Dockerfile.ci2 --cache-from ci2,$(CI2_LATEST),base,ubuntu:18.04 .

push: build
	docker tag ci2 $(CI2_LATEST)
	docker push $(CI2_LATEST)
	docker tag ci2 $(CI2_IMAGE)
	docker push $(CI2_IMAGE)

deploy: push
	sed -e "s,@sha@,$(shell git rev-parse --short=12 HEAD)," \
	  -e "s,@image@,$(CI2_IMAGE)," \
	  < deployment.yaml.in > deployment.yaml
	kubectl -n default apply -f deployment.yaml
else
build:
	make -C ../docker build
	docker build . -t ci2 -f Dockerfile.ci2
endif

run-local:
	PYTHONPATH=${PYTHONPATH:+${PYTHONPATH}:}../batch python3 ci/ci.py

run-docker:
	docker run -p 5000:5000 ci2
