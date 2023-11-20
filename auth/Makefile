include ../config.mk

.PHONY: build
build:
	$(MAKE) -C .. pushed-private-auth-image

.PHONY: deploy
deploy: build
	! [ -z $(NAMESPACE) ]  # call this like: make deploy NAMESPACE=default
	kubectl -n $(NAMESPACE) apply -f auth-driver-service-account.yaml
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"global":{"cloud":"$(CLOUD)"},"deploy":$(DEPLOY),"scope":"$(SCOPE)","default_ns":{"name":"$(NAMESPACE)"},"auth_image":{"image":"$(shell cat ../pushed-private-auth-image)"},"auth_database":{"user_secret_name":"sql-auth-user-config"}}' deployment.yaml deployment.yaml.out
	kubectl -n $(NAMESPACE) apply -f deployment.yaml.out

.PHONY:
clean:
	rm -f deployment.yaml.out
