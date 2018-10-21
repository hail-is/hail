.PHONY: hail.nginx.conf service.yaml build push deploy clean

PROJECT = $(shell gcloud config get-value project)
DOMAIN ?= hail.is
IP ?= 35.224.105.117

IMAGE = gcr.io/$(PROJECT)/gateway:$(shell docker images -q --no-trunc gateway | sed -e 's,[^:]*:,,')

hail.nginx.conf: hail.nginx.conf.in
	sed -e "s,@domain@,$(DOMAIN),g" \
	  < $< > $@

service.yaml: service.yaml.in
	sed -e "s,@ip@,$(IP),g" \
	  < $< > $@

deployment.yaml: build deployment.yaml.in
	sed -e "s,@sha@,$$(git rev-parse --short=12 HEAD),g" \
	  -e "s,@image@,$(IMAGE)," \
	  < $< > $@

build: Dockerfile hail.nginx.conf gateway.sh
	docker build -t gateway .

push: build
	docker tag gateway $(IMAGE)
	docker push $(IMAGE)

deploy: push service.yaml deployment.yaml
	kubectl apply -f service.yaml deployment.yaml

clean:
	rm -rf hail.nginx.conf service.yaml deployment.yaml
