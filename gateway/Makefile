.PHONY: push deploy clean

PROJECT = $(shell gcloud config get-value project)
DOMAIN ?= hail.is
IP ?= 35.224.105.117

GATEWAY_IMAGE = gcr.io/$(PROJECT)/gateway:$(shell docker images -q --no-trunc gateway | sed -e 's,[^:]*:,,')

STATIC_CONFIG = hail.nginx.conf service.yaml

$(STATIC_CONFIG): %: %.in
	sed -e "s,@domain@,$(DOMAIN),g" \
	  -e "s,@ip@,$(IP),g" \
	  < $< > $@

deployment.yaml: deployment.yaml.in build-gateway-stmp
	sed -e "s,@sha@,$$(git rev-parse --short=12 HEAD),g" \
	  -e "s,@image@,$(GATEWAY_IMAGE)," \
	  < $< > $@

build-gateway-stmp: Dockerfile hail.nginx.conf gateway.sh
	docker build -t gateway .
	touch build-gateway-stmp

push: build-gateway-stmp
	docker tag gateway $(IMAGE)
	docker push $(IMAGE)

deploy: push service.yaml deployment.yaml
	kubectl apply -f service.yaml deployment.yaml

clean:
	rm -rf $(STATIC_CONFIG) deployment.yaml build-gateway-stmp
