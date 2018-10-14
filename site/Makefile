.PHONY: build build-site build-run-letsencrypt push push-site push-run-letsencrypt \
  run-letsencrypt serve deploy-site

PROJECT = $(shell gcloud config get-value project)
DOMAIN ?= hail.is
IP ?= 35.224.105.117

STATIC_TEMPLATES = hail.nginx.conf.in run-letsencrypt-pod.yaml.in run-letsencrypt.nginx.conf.in service.yaml.in
$(STATIC_TEMPLATES): %: %.in
	sed -e "s,@project@,$(PROJECT),g" \
	  -e "s,@domain@,$(DOMAIN),g" \
	  -e "s,@ip@,$(IP),g" \
	  < $< > $@

# specialize
site-deployment.yaml: %: %.in
	sed -e "s,@sha@,$$(git rev-parse --short=12 HEAD),g" \
	  -e "s,@image@,$$(cat site-image)," \
	  < $< > $@

build: build-site build-run-letsencrypt

build-site: hail.nginx.conf
	docker build -t site .

build-run-letsencrypt: run-letsencrypt.nginx.conf
	docker build -t run-letsencrypt . -f Dockerfile.run-letsencrypt

push: push-site push-run-letsencrypt

push-site: IMAGE = gcr.io/$(PROJECT)/site:$(shell docker images -q --no-trunc site | sed -e 's,[^:]*:,,')
push-site: build-site
	echo $(IMAGE) > site-image
	docker tag site $(IMAGE)
	docker push $(IMAGE)

push-run-letsencrypt: build-run-letsencrypt
	docker tag run-letsencrypt gcr.io/$(PROJECT)/run-letsencrypt
	docker push gcr.io/$(PROJECT)/run-letsencrypt

run-letsencrypt: run-letsencrypt-pod.yaml service.yaml
	/bin/bash run-letsencrypt.sh

serve:
	docker run -it -p 80:80 -p 443:443 -v $$(pwd)/letsencrypt:/etc/letsencrypt site

deploy-site: service.yaml site-deployment.yaml
	kubectl apply -f service.yaml
	kubectl delete --ignore-not-found=true deployment site-deployment
	sleep 5
	kubectl create -f site-deployment.yaml
