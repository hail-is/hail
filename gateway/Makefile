include ../config.mk

.PHONY: deploy

IP := $(shell kubectl get secret global-config --template={{.data.ip}} | base64 --decode)

deploy:
	DOMAIN=$(DOMAIN) python3 ../ci/ci/envoy.py gateway ${HAIL}/letsencrypt/subdomains.txt ${HAIL}/gateway/cds.yaml ${HAIL}/gateway/rds.yaml
	kubectl -n default apply -f envoy.yaml
	kubectl -n default create configmap gateway-xds-config \
		--from-file=cds.yaml \
		--from-file=rds.yaml \
		--dry-run=client -o yaml \
		| kubectl apply -f -
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"global":{"ip":"$(IP)"}}' service.yaml service.yaml.out
	kubectl -n default apply -f service.yaml.out
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"global":{"docker_prefix":"$(DOCKER_PREFIX)"}}' deployment.yaml deployment.yaml.out
	kubectl -n default apply -f deployment.yaml.out
