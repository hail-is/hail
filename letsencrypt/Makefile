include ../config.mk

LETSENCRYPT_LATEST = gcr.io/$(PROJECT)/letsencrypt:latest
LETSENCRYPT_IMAGE = gcr.io/$(PROJECT)/letsencrypt:$(shell docker images -q --no-trunc letsencrypt | sed -e 's,[^:]*:,,')

STATIC_CONFIG = letsencrypt-pod.yaml letsencrypt.sh

$(STATIC_CONFIG): %: %.in
	sed -e "s,@project@,$(PROJECT),g" \
	  -e "s;@domains@;$(shell paste -s -d, domains.txt);g" \
	  -e "s,@domain@,$(DOMAIN),g" \
	  -e "s,@ip@,$(IP),g" \
	  < $< > $@

.PHONY: build
build: letsencrypt.sh
	$(MAKE) -C ../docker hail-ubuntu
	python3 ../ci/jinja2_render.py '{"hail_ubuntu_image":{"image":"hail-ubuntu"}}' Dockerfile Dockerfile.out
	docker build -f Dockerfile.out -t letsencrypt --cache-from letsencrypt,$(LETSENCRYPT_LATEST),hail-ubuntu .

.PHONY: push
push: build
	docker tag letsencrypt $(LETSENCRYPT_LATEST)
	docker push $(LETSENCRYPT_LATEST)
	docker tag letsencrypt $(LETSENCRYPT_IMAGE)
	docker push $(LETSENCRYPT_IMAGE)

.PHONY: run
run: letsencrypt-pod.yaml service.yaml push
	kubectl -n default apply -f service.yaml
	python3 ../ci/jinja2_render.py '{"letsencrypt_image":{"image":"$(LETSENCRYPT_IMAGE)"}}' letsencrypt-pod.yaml letsencrypt-pod.yaml.out
	/bin/bash run-letsencrypt.sh

.PHONY: clean
clean:
	rm -rf $(STATIC_CONFIG)
