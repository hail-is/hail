include ../config.mk

ROUTER_LATEST = $(DOCKER_PREFIX)/router:latest
ROUTER_IMAGE = $(DOCKER_PREFIX)/router:$(shell docker images -q --no-trunc router | sed -e 's,[^:]*:,,')

.PHONY: build
build:
	$(MAKE) -C ../docker hail-ubuntu
	-docker pull $(ROUTER_LATEST)
	python3 ../ci/jinja2_render.py '{"hail_ubuntu_image":{"image":"hail-ubuntu"}}' Dockerfile Dockerfile.out
	docker build -f Dockerfile.out -t router --cache-from router,$(ROUTER_LATEST),hail-ubuntu .

.PHONY: push
push: build
	docker tag router $(ROUTER_LATEST)
	docker push $(ROUTER_LATEST)
	docker tag router $(ROUTER_IMAGE)
	docker push $(ROUTER_IMAGE)

.PHONY: deploy
deploy: push
	! [ -z $(NAMESPACE) ]  # call this like: make deploy NAMESPACE=default
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":$(DEPLOY),"router_image":{"image":"$(ROUTER_IMAGE)"},"default_ns":{"name":"$(NAMESPACE)"},"global":{"domain":"$(DOMAIN)"}}' deployment.yaml deployment.yaml.out
	kubectl -n $(NAMESPACE) apply -f deployment.yaml.out

.PHONY: clean
clean:
	rm -rf deployment.yaml
