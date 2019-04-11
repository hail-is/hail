.PHONY: build clean

ifeq ($(IN_HAIL_CI),1)
.PHONY: push deploy

PROJECT = $(shell gcloud config get-value project)
ROUTER_LATEST = gcr.io/$(PROJECT)/router:latest
ROUTER_IMAGE = gcr.io/$(PROJECT)/router:$(shell docker images -q --no-trunc router | sed -e 's,[^:]*:,,')
DOMAIN ?= hail.is

deployment.yaml: deployment.yaml.in build
	sed -e "s,@sha@,$$(git rev-parse --short=12 HEAD),g" \
	  -e "s,@image@,$(ROUTER_IMAGE)," \
	  -e "s,@domain@,$(DOMAIN)," \
	  < $< > $@

build:
	docker pull ubuntu:18.04
	-docker pull $(ROUTER_LATEST)
	docker build -t router --cache-from router,$(ROUTER_LATEST),ubuntu:18.04 .

push: build
	docker tag router $(ROUTER_LATEST)
	docker push $(ROUTER_LATEST)
	docker tag router $(ROUTER_IMAGE)
	docker push $(ROUTER_IMAGE)

deploy: push deployment.yaml
	kubectl -n default apply -f deployment.yaml
else
build:
	docker build -t router .
endif

clean:
	rm -rf deployment.yaml
