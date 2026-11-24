include ../config.mk

.PHONY: envoy-xds-config deploy

IP := $(shell kubectl get secret global-config --template={{.data.ip}} | base64 --decode)
SHA := $(shell git rev-parse --short=12 HEAD)
CLOUD := $(shell kubectl get secret global-config --template={{.data.cloud}} | base64 --decode)
HTTP_TARGET_GROUP_ARN = $(shell kubectl get secret global-config --template={{.data.gateway_http_target_group_arn}} | base64 --decode)
HTTPS_TARGET_GROUP_ARN = $(shell kubectl get secret global-config --template={{.data.gateway_https_target_group_arn}} | base64 --decode)

envoy-xds-config:
	python3 ../ci/ci/envoy.py gateway $(DOMAIN) ${HAIL}/letsencrypt/subdomains.txt ${HAIL}/gateway/cds.yaml.out ${HAIL}/gateway/rds.yaml.out
	kubectl -n default create configmap gateway-xds-config \
		--from-file=cds.yaml=cds.yaml.out \
		--from-file=rds.yaml=rds.yaml.out \
		--dry-run=client -o yaml \
		| kubectl apply -f -

deploy:
	python3 ../ci/jinja2_render.py '{"domain":"$(DOMAIN)"}' envoy.yaml envoy.yaml.out
	kubectl -n default apply -f envoy.yaml.out
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(SHA)"},"global":{"ip":"$(IP)","cloud":"$(CLOUD)"}}' service.yaml service.yaml.out
	kubectl -n default apply -f service.yaml.out
ifeq ($(CLOUD),aws)
    python3 ../ci/jinja2_render.py '{"global":{"gateway_http_target_group_arn":"$(HTTP_TARGET_GROUP_ARN)","gateway_https_target_group_arn":"$(HTTPS_TARGET_GROUP_ARN)"}}' target-group-bindings.yaml target-group-bindings.yaml.out
    kubectl -n default apply -f target-group-bindings.yaml.out
endif
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(SHA)"},"global":{"docker_prefix":"$(DOCKER_PREFIX)"}}' deployment.yaml deployment.yaml.out
	kubectl -n default apply -f deployment.yaml.out
