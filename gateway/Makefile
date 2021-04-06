include ../config.mk

.PHONY: build push deploy

GATEWAY_LATEST = $(DOCKER_PREFIX)/gateway:latest
GATEWAY_IMAGE = $(DOCKER_PREFIX)/gateway:$(shell docker images -q --no-trunc gateway | sed -e 's,[^:]*:,,')

build:
	$(MAKE) -C ../docker hail-ubuntu
	-docker pull $(GATEWAY_LATEST)
	python3 ../ci/jinja2_render.py '{"hail_ubuntu_image":{"image":"hail-ubuntu"}}' Dockerfile Dockerfile.out
	python3 ../ci/jinja2_render.py '{"domain": "$(DOMAIN)","subdomains":[$(shell sed -e 's/\(.*\)/"\1"/' ../letsencrypt/subdomains.txt | paste -s -d, -)]}' gateway.nginx.conf gateway.nginx.conf.out
	docker build -t gateway -f Dockerfile.out --cache-from gateway,$(GATEWAY_LATEST),hail-ubuntu .

push: build
	docker tag gateway $(GATEWAY_LATEST)
	docker push $(GATEWAY_LATEST)
	docker tag gateway $(GATEWAY_IMAGE)
	docker push $(GATEWAY_IMAGE)

deploy: push
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":$(DEPLOY),"global":{"ip":"$(IP)"}}' service.yaml service.yaml.out
	kubectl -n default apply -f service.yaml.out
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":$(DEPLOY),"gateway_image":{"image":"$(GATEWAY_IMAGE)"},"global":{"project":"$(PROJECT)"}}' deployment.yaml deployment.yaml.out
	kubectl -n default apply -f deployment.yaml.out
