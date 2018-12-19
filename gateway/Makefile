.PHONY: Dockerfile hail.nginx.conf service.yaml deployment.yaml build push deploy clean

PROJECT = $(shell gcloud config get-value project)
DOMAIN ?= hail.is
IP ?= 35.224.105.117

LETSENCRYPT_ONLY = 0

ifeq ($(LETSENCRYPT_ONLY),1)
NGINX_CONF = letsencrypt.nginx.conf
else
NGINX_CONF = hail.nginx.conf
endif

IMAGE = gcr.io/$(PROJECT)/gateway:$(shell docker images -q --no-trunc gateway | sed -e 's,[^:]*:,,')

Dockerfile: Dockerfile.in
	sed -e "s,@nginx_conf@,$(NGINX_CONF),g" \
	  < $< > $@

hail.nginx.conf: hail.nginx.conf.in
	sed -e "s,@domain@,$(DOMAIN),g" \
	  < $< > $@

service.yaml: service.yaml.in
	sed -e "s,@ip@,$(IP),g" \
	  < $< > $@

deployment.yaml: deployment.yaml.in build
	sed -e "s,@sha@,$$(git rev-parse --short=12 HEAD),g" \
	  -e "s,@image@,$(IMAGE)," \
	  < $< > $@

build: Dockerfile $(NGINX_CONF)
	docker build -t gateway .

push: build
	docker tag gateway $(IMAGE)
	docker push $(IMAGE)

deploy: push service.yaml deployment.yaml
	kubectl -n default apply -f service.yaml
	kubectl -n default apply -f deployment.yaml

clean:
	rm -rf hail.nginx.conf service.yaml deployment.yaml
