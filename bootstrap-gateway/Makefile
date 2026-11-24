include ../config.mk

.PHONY: build deploy

GATEWAY_IMAGE := $(DOCKER_PREFIX)/gateway:$(TOKEN)
IP := $(shell kubectl get secret global-config --template={{.data.ip}} | base64 --decode)
CLOUD := $(shell kubectl get secret global-config --template={{.data.cloud}} | base64 --decode)
HTTP_TARGET_GROUP_ARN = $(shell kubectl get secret global-config --template={{.data.gateway_http_target_group_arn}} | base64 --decode)
HTTPS_TARGET_GROUP_ARN = $(shell kubectl get secret global-config --template={{.data.gateway_https_target_group_arn}} | base64 --decode)

build:
	$(MAKE) -C .. hail-ubuntu-image
	python3 ../ci/jinja2_render.py '{"hail_ubuntu_image":{"image":"'$$(cat ../hail-ubuntu-image)'"}}' Dockerfile Dockerfile.out
	../docker-build.sh . Dockerfile.out $(GATEWAY_IMAGE)
	docker push $(GATEWAY_IMAGE)

deploy: build
    python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":true,"global":{"ip":"$(IP)","cloud":"$(CLOUD)"}}' service.yaml service.yaml.out
	kubectl -n default apply -f service.yaml.out
ifeq ($(CLOUD),aws)
    python3 ../ci/jinja2_render.py '{"global":{"gateway_http_target_group_arn":"$(HTTP_TARGET_GROUP_ARN)","gateway_https_target_group_arn":"$(HTTPS_TARGET_GROUP_ARN)"}}' ../gateway/target-group-bindings.yaml target-group-bindings.yaml.out
    kubectl -n default apply -f target-group-bindings.yaml.out
endif
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"gateway_image":{"image":"$(GATEWAY_IMAGE)"}}' deployment.yaml deployment.yaml.out
	kubectl -n default apply -f deployment.yaml.out
