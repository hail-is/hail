.PHONY: push-gateway push-run-letsencrypt \
  run-letsencrypt deploy-gateway clean

PROJECT = $(shell gcloud config get-value project)
DOMAIN ?= hail.is
IP ?= 35.224.105.117

GATEWAY_IMAGE = gcr.io/$(PROJECT)/gateway:$(shell docker images -q --no-trunc gateway | sed -e 's,[^:]*:,,')
RUN_LETSENCRYPT_IMAGE = gcr.io/$(PROJECT)/run-letsencrypt:$(shell docker images -q --no-trunc run-letsencrypt | sed -e 's,[^:]*:,,')

STATIC_CONFIG = hail.nginx.conf run-letsencrypt-pod.yaml run-letsencrypt.nginx.conf service.yaml

$(STATIC_CONFIG): %: %.in
	sed -e "s,@project@,$(PROJECT),g" \
	  -e "s,@domain@,$(DOMAIN),g" \
	  -e "s,@ip@,$(IP),g" \
	  < $< > $@

gateway-deployment.yaml: gateway-deployment.yaml.in build-gateway-stmp
	sed -e "s,@sha@,$$(git rev-parse --short=12 HEAD),g" \
	  -e "s,@image@,$(GATEWAY_IMAGE)," \
	  < $< > $@

build-gateway-stmp: Dockerfile hail.nginx.conf gateway.sh
	docker build -t gateway .
	touch build-gateway-stmp

build-run-letsencrypt-stmp: Dockerfile.run-letsencrypt run-letsencrypt.nginx.conf
	docker build -t run-letsencrypt . -f Dockerfile.run-letsencrypt
	touch build-run-letsencrypt-stmp

push-gateway: build-gateway-stmp
	docker tag gateway $(IMAGE)
	docker push $(IMAGE)

push-run-letsencrypt: build-run-letsencrypt-stmp
	docker tag run-letsencrypt $(RUN_LETSENCRYPT_IMAGE)
	docker push $(RUN_LETSENCRYPT_IMAGE)

run-letsencrypt: push-run-letsencrypt run-letsencrypt-pod.yaml service.yaml
	/bin/bash run-letsencrypt.sh

deploy-gateway: build-gateway-stmp service.yaml gateway-deployment.yaml
	kubectl apply -f service.yaml
	kubectl delete --ignore-not-found=true deployment gateway-deployment
	sleep 5
	kubectl create -f gateway-deployment.yaml

clean:
	rm -rf $(STATIC_CONFIG) gateway-deployment.yaml
	rm -rf build-gateway-stmp build-run-letsencrypt-stmp
