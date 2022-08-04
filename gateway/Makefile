include ../config.mk

.PHONY: deploy

IP := $(shell kubectl get secret global-config --template={{.data.ip}} | base64 --decode)

deploy: build
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"global":{"ip":"$(IP)"}}' service.yaml service.yaml.out
	kubectl -n default apply -f service.yaml.out
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"global":{"docker_prefix":"$(DOCKER_PREFIX)","domain":"$(DOMAIN)"},"subdomains":[$(shell sed -e 's/\(.*\)/"\1"/' ../letsencrypt/subdomains.txt | paste -s -d, -)]}' deployment.yaml deployment.yaml.out
	kubectl -n default apply -f deployment.yaml.out
