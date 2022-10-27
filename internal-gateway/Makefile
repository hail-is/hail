include ../config.mk

.PHONY: envoy-xds-config deploy

INTERNAL_IP := $(shell kubectl get secret global-config --template={{.data.internal_ip}} | base64 --decode)

envoy-xds-config:
	HAIL_DOMAIN=$(DOMAIN) python3 ../ci/ci/envoy.py internal-gateway ${HAIL}/letsencrypt/subdomains.txt ${HAIL}/internal-gateway/cds.yaml.out ${HAIL}/internal-gateway/rds.yaml.out
	kubectl -n default create configmap internal-gateway-xds-config \
		--from-file=cds.yaml=cds.yaml.out \
		--from-file=rds.yaml=rds.yaml.out \
		--dry-run=client -o yaml \
		| kubectl apply -f -

deploy:
	kubectl -n default apply -f envoy.yaml
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"global":{"internal_ip":"$(INTERNAL_IP)"}}' service.yaml service.yaml.out
	kubectl -n default apply -f service.yaml.out
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"global":{"docker_prefix":"$(DOCKER_PREFIX)"}}' deployment.yaml deployment.yaml.out
	kubectl -n default apply -f deployment.yaml.out
