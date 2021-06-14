include ../config.mk

TOKEN = $(shell cat /dev/urandom | LC_ALL=C tr -dc 'a-z0-9' | head -c 12)

LETSENCRYPT_IMAGE := $(DOCKER_PREFIX)/letsencrypt:$(TOKEN)

.PHONY: build start-service run clean

build:
	../docker-build.sh . Dockerfile $(LETSENCRYPT_IMAGE)

start-service:
	kubectl -n default apply -f service.yaml

DRY_RUN ?= false
run: build
	echo $(DOMAIN) > domains.txt.out
	echo internal.$(DOMAIN) >> domains.txt.out
	sed 's/$$/.$(DOMAIN)/g' subdomains.txt >> domains.txt.out
	python3 ../ci/jinja2_render.py '{"letsencrypt_image":{"image":"$(LETSENCRYPT_IMAGE)"},"domain":"$(DOMAIN)","domains":"'$$(paste -s -d, domains.txt.out)'","dry_run":$(DRY_RUN)}' letsencrypt-pod.yaml letsencrypt-pod.yaml.out
	/bin/bash run-letsencrypt.sh

.PHONY: clean
clean:
	rm -rf $(STATIC_CONFIG)
