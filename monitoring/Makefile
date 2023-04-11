include ../config.mk

.PHONY: build
build:
	$(MAKE) -C .. monitoring-image

.PHONY: deploy
deploy: build
	! [ -z $(NAMESPACE) ]  # call this like: make deploy NAMESPACE=default
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":$(DEPLOY),"default_ns":{"name":"$(NAMESPACE)"},"monitoring_image":{"image":"'$$(cat ../monitoring-image)'"},"monitoring_database":{"user_secret_name":"sql-monitoring-user-config"},"scope":"$(SCOPE)"}' deployment.yaml deployment.yaml.out
	kubectl -n $(NAMESPACE) apply -f deployment.yaml.out

.PHONY: clean
clean:
	rm -f Dockerfile.out deployment.yaml.out
