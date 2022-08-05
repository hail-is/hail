include ../config.mk

.PHONY: deploy

INTERNAL_IP := $(shell kubectl get secret global-config --template={{.data.internal_ip}} | base64 --decode)

deploy:
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"global":{"internal_ip":"$(INTERNAL_IP)"}}' service.yaml service.yaml.out
	kubectl -n default apply -f service.yaml.out
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"global":{"docker_prefix":"$(DOCKER_PREFIX)","domain":"$(DOMAIN)"},"subdomains":[$(shell sed -e 's/\(.*\)/"\1"/' ../letsencrypt/subdomains.txt | paste -s -d, -)],"internal_namespaces":[$(shell sed -e 's/\(.*\)/"\1"/' ../internal-namespaces.txt | paste -s -d, -)]}' deployment.yaml deployment.yaml.out
	kubectl -n default apply -f deployment.yaml.out
