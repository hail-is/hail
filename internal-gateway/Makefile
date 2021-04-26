include ../config.mk

.PHONY: build push deploy

INTERNAL_GATEWAY_LATEST = $(DOCKER_PREFIX)/internal-gateway:latest
INTERNAL_GATEWAY_IMAGE = $(DOCKER_PREFIX)/internal-gateway:$(shell docker images -q --no-trunc internal-gateway | sed -e 's,[^:]*:,,')

build:
	$(MAKE) -C ../docker hail-ubuntu
	-docker pull $(INTERNAL_GATEWAY_LATEST)
	python3 ../ci/jinja2_render.py '{"hail_ubuntu_image":{"image":"hail-ubuntu"}}' Dockerfile Dockerfile.out
	docker build -t internal-gateway -f Dockerfile.out --cache-from internal-gateway,$(INTERNAL_GATEWAY_LATEST),hail-ubuntu .

push: build
	docker tag internal-gateway $(INTERNAL_GATEWAY_LATEST)
	docker push $(INTERNAL_GATEWAY_LATEST)
	docker tag internal-gateway $(INTERNAL_GATEWAY_IMAGE)
	docker push $(INTERNAL_GATEWAY_IMAGE)

deploy: push
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":$(DEPLOY),"global":{"internal_ip":"$(INTERNAL_IP)"}}' service.yaml service.yaml.out
	kubectl -n default apply -f service.yaml.out
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":$(DEPLOY),"internal_gateway_image":{"image":"$(INTERNAL_GATEWAY_IMAGE)"}}' deployment.yaml deployment.yaml.out
	kubectl -n default apply -f deployment.yaml.out
