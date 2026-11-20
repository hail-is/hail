include ../config.mk

.PHONY: envoy-xds-config deploy

INTERNAL_IP := $(shell kubectl get secret global-config --template={{.data.internal_ip}} | base64 --decode)
CLOUD := $(shell kubectl get secret global-config --template={{.data.cloud}} | base64 --decode)
TARGET_GROUP_ARN = $(shell kubectl get secret global-config --template={{.data.internal_gateway_target_group_arn}} | base64 --decode)

envoy-xds-config:
	python3 ../ci/ci/envoy.py internal-gateway $(DOMAIN) ${HAIL}/letsencrypt/subdomains.txt ${HAIL}/internal-gateway/cds.yaml.out ${HAIL}/internal-gateway/rds.yaml.out
	kubectl -n default create configmap internal-gateway-xds-config \
		--from-file=cds.yaml=cds.yaml.out \
		--from-file=rds.yaml=rds.yaml.out \
		--dry-run=client -o yaml \
		| kubectl apply -f -

deploy:
	kubectl -n default apply -f envoy.yaml
    python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"global":{"internal_ip":"$(INTERNAL_IP)","cloud":"$(CLOUD)"}}' service.yaml service.yaml.out
	kubectl -n default apply -f service.yaml.out
ifeq ($(CLOUD),aws)
     python3 ../ci/jinja2_render.py '{"global":{"internal_gateway_target_group_arn":"$(TARGET_GROUP_ARN)"}}' target-group-bindings.yaml target-group-bindings.yaml.out
     kubectl -n default apply -f target-group-bindings.yaml.out
endif
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"global":{"docker_prefix":"$(DOCKER_PREFIX)"}}' deployment.yaml deployment.yaml.out
	kubectl -n default apply -f deployment.yaml.out
