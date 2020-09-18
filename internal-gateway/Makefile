include ../config.mk

.PHONY: build push deploy

INTERNAL_GATEWAY_LATEST = gcr.io/$(PROJECT)/internal-gateway:latest
INTERNAL_GATEWAY_IMAGE = gcr.io/$(PROJECT)/internal-gateway:$(shell docker images -q --no-trunc internal-gateway | sed -e 's,[^:]*:,,')

build:
	docker pull bitnami/minideb:buster
	-docker pull $(INTERNAL_GATEWAY_LATEST)
	docker build -t internal-gateway --cache-from internal-gateway,$(INTERNAL_GATEWAY_LATEST),bitnami/minideb:buster .

push: build
	docker tag internal-gateway $(INTERNAL_GATEWAY_LATEST)
	docker push $(INTERNAL_GATEWAY_LATEST)
	docker tag internal-gateway $(INTERNAL_GATEWAY_IMAGE)
	docker push $(INTERNAL_GATEWAY_IMAGE)

deploy: push
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":true,"global":{"internal_ip":"$(INTERNAL_IP)"}}' service.yaml service.yaml.out
	kubectl -n default apply -f service.yaml.out
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":true,"internal_gateway_image":{"image":"$(INTERNAL_GATEWAY_IMAGE)"}}' deployment.yaml deployment.yaml.out
	kubectl -n default apply -f deployment.yaml.out
