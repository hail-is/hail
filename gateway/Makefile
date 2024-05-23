include ../config.mk

.PHONY: envoy-xds-config deploy

IP := $(shell kubectl get secret global-config --template={{.data.ip}} | base64 --decode)
SHA := $(shell git rev-parse --short=12 HEAD)

envoy-xds-config:
	HAIL_PRODUCTION_DOMAIN=$(DOMAIN) python3 ../ci/ci/envoy.py gateway ${HAIL}/letsencrypt/subdomains.txt ${HAIL}/gateway/cds.yaml.out ${HAIL}/gateway/rds.yaml.out
	kubectl -n default create configmap gateway-xds-config \
		--from-file=cds.yaml=cds.yaml.out \
		--from-file=rds.yaml=rds.yaml.out \
		--dry-run=client -o yaml \
		| kubectl apply -f -

deploy:
	python3 ../ci/jinja2_render.py '{"domain":"$(DOMAIN)"}' envoy.yaml envoy.yaml.out
	kubectl -n default apply -f envoy.yaml.out
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(SHA)"},"global":{"ip":"$(IP)"}}' service.yaml service.yaml.out
	kubectl -n default apply -f service.yaml.out
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(SHA)"},"global":{"docker_prefix":"$(DOCKER_PREFIX)"}}' deployment.yaml deployment.yaml.out
	kubectl -n default apply -f deployment.yaml.out
