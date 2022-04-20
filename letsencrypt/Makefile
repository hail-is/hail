include ../config.mk

TOKEN = $(shell cat /dev/urandom | LC_ALL=C tr -dc 'a-z0-9' | head -c 12)

LETSENCRYPT_IMAGE := $(DOCKER_PREFIX)/letsencrypt:$(TOKEN)

.PHONY: build start-service run clean

build:
	../docker-build.sh . Dockerfile $(LETSENCRYPT_IMAGE)

DRY_RUN ?= false
run: build
	echo $(DOMAIN) > domains.txt.out
	echo internal.$(DOMAIN) >> domains.txt.out
	sed 's/$$/.$(DOMAIN)/g' subdomains.txt >> domains.txt.out
	python3 ../ci/jinja2_render.py '{"letsencrypt_image":{"image":"$(LETSENCRYPT_IMAGE)"},"domain":"$(DOMAIN)","domains":"'$$(paste -s -d, domains.txt.out)'","dry_run":$(DRY_RUN)}' letsencrypt-pod.yaml letsencrypt-pod.yaml.out
	/bin/bash run-letsencrypt.sh letsencrypt-pod.yaml.out

revoke: build
	! [ -z "$(CERT_IDS_TO_REVOKE)" ]  # call this like: make deploy CERT_IDS_TO_REVOKE='abc123 def567'
	python3 ../ci/jinja2_render.py '{"letsencrypt_image":{"image":"$(LETSENCRYPT_IMAGE)"},"cert_ids_to_revoke":"$(CERT_IDS_TO_REVOKE)"}' revoke-certs-pod.yaml revoke-certs-pod.yaml.out
	/bin/bash run-letsencrypt.sh revoke-certs-pod.yaml.out

.PHONY: clean
clean:
	rm -rf $(STATIC_CONFIG)
