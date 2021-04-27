include ../config.mk

LETSENCRYPT_LATEST = $(DOCKER_PREFIX)/letsencrypt:latest
LETSENCRYPT_IMAGE = $(DOCKER_PREFIX)/letsencrypt:$(shell docker images -q --no-trunc letsencrypt | sed -e 's,[^:]*:,,')

.PHONY: build push start-service run clean

build:
	docker build -f Dockerfile -t letsencrypt --cache-from letsencrypt,$(LETSENCRYPT_LATEST) .

.PHONY: push
push: build
	docker tag letsencrypt $(LETSENCRYPT_LATEST)
	docker push $(LETSENCRYPT_LATEST)
	docker tag letsencrypt $(LETSENCRYPT_IMAGE)
	docker push $(LETSENCRYPT_IMAGE)

start-service:
	kubectl -n default apply -f service.yaml

DRY_RUN ?= false
run: push
	echo $(DOMAIN) > domains.txt.out
	echo internal.$(DOMAIN) >> domains.txt.out
	sed 's/$$/.$(DOMAIN)/g' subdomains.txt >> domains.txt.out
	python3 ../ci/jinja2_render.py '{"letsencrypt_image":{"image":"$(LETSENCRYPT_IMAGE)"},"domain":"$(DOMAIN)","domains":"'$$(paste -s -d, domains.txt.out)'","dry_run":$(DRY_RUN)}' letsencrypt-pod.yaml letsencrypt-pod.yaml.out
	/bin/bash run-letsencrypt.sh

.PHONY: clean
clean:
	rm -rf $(STATIC_CONFIG)
