include ../config.mk

.PHONY: build
build:
	$(MAKE) -C .. push-private-ci-image push-private-ci-utils-image push-private-hail-buildkit-image

.PHONY: deploy
deploy: build
	! [ -z $(NAMESPACE) ]  # call this like: make deploy NAMESPACE=default
	python3 jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"global":{"cloud":"$(CLOUD)"},"deploy":$(DEPLOY),"scope":"$(SCOPE)","default_ns":{"name":"$(NAMESPACE)"},"ci_image":{"image":"$(shell cat ../ci-image)"},"ci_utils_image":{"image":"'$$(cat ../ci-utils-image)'"},"ci_database":{"user_secret_name":"sql-ci-user-config"},"hail_buildkit_image":{"image":"'$$(cat ../hail-buildkit-image)'"}}' deployment.yaml deployment.yaml.out
	kubectl -n $(NAMESPACE) apply -f deployment.yaml.out
