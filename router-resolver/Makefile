.PHONY: build

ifeq ($(IN_HAIL_CI),1)
.PHONY: push deploy

PROJECT = $(shell gcloud config get-value project)
ROUTER_RESOLVER_LATEST = gcr.io/$(PROJECT)/router-resolver:latest
ROUTER_RESOLVER_IMAGE = gcr.io/$(PROJECT)/router-resolver:$(shell docker images -q --no-trunc router-resolver:latest | sed -e 's,[^:]*:,,')

build:
	make -C ../docker build
	-docker pull $(ROUTER_RESOLVER_LATEST)
	docker build -t router-resolver --cache-from router-resolver,$(ROUTER_RESOLVER_LATEST),base,ubuntu:18.04 .

push: build
	docker tag router-resolver $(ROUTER_RESOLVER_LATEST)
	docker push $(ROUTER_RESOLVER_LATEST)
	docker tag router-resolver $(ROUTER_RESOLVER_IMAGE)
	docker push $(ROUTER_RESOLVER_IMAGE)

deploy: push
	sed -e "s,@sha@,$(shell git rev-parse --short=12 HEAD)," \
	  -e "s,@image@,$(ROUTER_RESOLVER_IMAGE)," \
	  < deployment.yaml.in > deployment.yaml
	kubectl -n default apply -f deployment.yaml
else
build:
	make -C ../docker build
	docker build -t router-resolver .
endif
