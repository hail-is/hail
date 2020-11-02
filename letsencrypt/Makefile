include ../config.mk

STATIC_CONFIG = letsencrypt-pod.yaml letsencrypt.sh

.PHONY: $(STATIC_CONFIG) build push start-service run clean

$(STATIC_CONFIG): %: %.in
	sed -e "s,@project@,$(PROJECT),g" \
	  -e "s;@domains@;$(shell paste -s -d, domains.txt);g" \
	  -e "s,@domain@,$(DOMAIN),g" \
	  -e "s,@ip@,$(IP),g" \
	  < $< > $@

build: letsencrypt.sh
	python3 ../ci/jinja2_render.py '{"global":{"project":"$(PROJECT)"}}' Dockerfile Dockerfile.out
	docker build -f Dockerfile.out -t letsencrypt .

push: build
	docker tag letsencrypt gcr.io/$(PROJECT)/letsencrypt
	docker push gcr.io/$(PROJECT)/letsencrypt

start-service: service.yaml
	kubectl -n default apply -f service.yaml

run: letsencrypt-pod.yaml service.yaml push
	/bin/bash run-letsencrypt.sh

clean:
	rm -rf $(STATIC_CONFIG)
