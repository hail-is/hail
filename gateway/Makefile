.PHONY: build clean

PROJECT = $(shell gcloud config get-value project)
GATEWAY_LATEST = gcr.io/$(PROJECT)/gateway:latest
GATEWAY_IMAGE = gcr.io/$(PROJECT)/gateway:$(shell docker images -q --no-trunc gateway | sed -e 's,[^:]*:,,')
DOMAIN ?= hail.is
IP ?= 35.224.105.117

service.yaml: service.yaml.in
	sed -e "s,@ip@,$(IP),g" \
	  < $< > $@

deployment.yaml: deployment.yaml.in build
	sed -e "s,@sha@,$$(git rev-parse --short=12 HEAD),g" \
	  -e "s,@image@,$(GATEWAY_IMAGE)," \
	  < $< > $@

ifeq ($(IN_HAIL_CI),1)
.PHONY: push deploy
build:
	docker pull ubuntu:18.04
	-docker pull $(GATEWAY_LATEST)
	docker build -t gateway --cache-from gateway,$(GATEWAY_LATEST),ubuntu:18.04 .

push: build
	docker tag gateway $(GATEWAY_LATEST)
	docker push $(GATEWAY_LATEST)
	docker tag gateway $(GATEWAY_IMAGE)
	docker push $(GATEWAY_IMAGE)

deploy: push service.yaml deployment.yaml
	kubectl -n default apply -f service.yaml
	kubectl -n default apply -f deployment.yaml
else
build:
	docker build -t gateway .
endif

clean:
	rm -rf service.yaml deployment.yaml
