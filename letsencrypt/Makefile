include ../config.mk

.PHONY: build start-service run clean

build:
	$(MAKE) -C .. pushed-private-letsencrypt-image

DRY_RUN ?= false
run: build
	echo $(DOMAIN) > domains.txt.out
	echo internal.$(DOMAIN) >> domains.txt.out
	sed 's/$$/.$(DOMAIN)/g' subdomains.txt >> domains.txt.out
	python3 ../ci/jinja2_render.py '{"letsencrypt_image":{"image":"$(shell cat ../pushed-private-letsencrypt-image)"},"domain":"$(DOMAIN)","domains":"'$$(paste -s -d, domains.txt.out)'","dry_run":$(DRY_RUN)}' letsencrypt-pod.yaml letsencrypt-pod.yaml.out
	/bin/bash run-letsencrypt.sh letsencrypt-pod.yaml.out

revoke: build
	! [ -z "$(CERT_IDS_TO_REVOKE)" ]  # call this like: make deploy CERT_IDS_TO_REVOKE='abc123 def567'
	python3 ../ci/jinja2_render.py '{"letsencrypt_image":{"image":"$(shell cat ../pushed-private-letsencrypt-image)"},"cert_ids_to_revoke":"$(CERT_IDS_TO_REVOKE)"}' revoke-certs-pod.yaml revoke-certs-pod.yaml.out
	/bin/bash run-letsencrypt.sh revoke-certs-pod.yaml.out

.PHONY: clean
clean:
	rm -rf $(STATIC_CONFIG)
