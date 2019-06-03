.PHONY: build push deploy

PROJECT = $(shell gcloud config get-value project)
MONITORING_ROUTER_LATEST = gcr.io/$(PROJECT)/monitoring-router:latest
MONITORING_ROUTER_IMAGE = gcr.io/$(PROJECT)/monitoring-router:$(shell docker images -q --no-trunc monitoring-router | sed -e 's,[^:]*:,,')

build:
	docker pull ubuntu:18.04
	-docker pull $(MONITORING_ROUTER_LATEST)
	docker build -t monitoring-router --cache-from monitoring-router,$(MONITORING_ROUTER_LATEST),ubuntu:18.04 .

push: build
	docker tag monitoring-router $(MONITORING_ROUTER_LATEST)
	docker push $(MONITORING_ROUTER_LATEST)
	docker tag monitoring-router $(MONITORING_ROUTER_IMAGE)
	docker push $(MONITORING_ROUTER_IMAGE)

# FIXME use stateful set or prometheus, grafana
deploy: push
	kubectl -n monitoring delete deployments prometheus grafana
	sleep 5
	kubectl apply -f monitoring.yaml
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":true,"monitoring_router_image":{"image":"$(MONITORING_ROUTER_IMAGE)"}}' router.yaml router.yaml.out
	kubectl apply -f router.yaml.out
