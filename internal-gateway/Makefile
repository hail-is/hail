include ../config.mk

.PHONY: deploy envoy_config

INTERNAL_IP := $(shell kubectl get secret global-config --template={{.data.internal_ip}} | base64 --decode)

envoy_config:
	python3 ../ci/jinja2_render.py '{"global":{"domain":"$(DOMAIN)"},"subdomains":[$(shell sed -e 's/\(.*\)/"\1"/' ../letsencrypt/subdomains.txt | paste -s -d, -)],"internal_namespaces":[$(shell sed -e 's/\(.*\)/"\1"/' ../internal-namespaces.txt | paste -s -d, -)]}' envoy.yaml envoy.yaml.out
	kubectl -n default apply -f envoy.yaml.out

deploy:
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"global":{"internal_ip":"$(INTERNAL_IP)"}}' service.yaml service.yaml.out
	kubectl -n default apply -f service.yaml.out
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"global":{"docker_prefix":"$(DOCKER_PREFIX)"}}' deployment.yaml deployment.yaml.out
	kubectl -n default apply -f deployment.yaml.out
