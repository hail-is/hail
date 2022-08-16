include ../config.mk

.PHONY: envoy-config deploy

IP := $(shell kubectl get secret global-config --template={{.data.ip}} | base64 --decode)

envoy-config:
	HAIL_DOMAIN=$(DOMAIN) python3 ../ci/ci/envoy.py gateway ${HAIL}/letsencrypt/subdomains.txt ${HAIL}/gateway/cds.yaml ${HAIL}/gateway/rds.yaml
	python3 ../ci/jinja2_render.py '{"domain":"$(DOMAIN)"}' envoy.yaml envoy.yaml.out
	kubectl -n default apply -f envoy.yaml.out
	kubectl -n default create configmap gateway-xds-config \
		--from-file=cds.yaml \
		--from-file=rds.yaml \
		--dry-run=client -o yaml \
		| kubectl apply -f -

deploy: envoy-config
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"global":{"ip":"$(IP)"}}' service.yaml service.yaml.out
	kubectl -n default apply -f service.yaml.out
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"global":{"docker_prefix":"$(DOCKER_PREFIX)"}}' deployment.yaml deployment.yaml.out
	kubectl -n default apply -f deployment.yaml.out
