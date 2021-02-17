include ../config.mk

AUTH_LATEST = $(DOCKER_PREFIX)/auth:latest
AUTH_IMAGE = $(DOCKER_PREFIX)/auth:$(shell docker images -q --no-trunc auth:latest | sed -e 's,[^:]*:,,')

EXTRA_PYTHONPATH := ../hail/python:../gear:../web_common
PYTHON := PYTHONPATH=$${PYTHONPATH:+$${PYTHONPATH}:}$(EXTRA_PYTHONPATH) python3

.PHONY: check
check:
	$(PYTHON) -m flake8 auth
	$(PYTHON) -m pylint --rcfile ../pylintrc auth --score=n
	../check-sql.sh

.PHONY: build
build:
	$(MAKE) -C ../docker build
	-docker pull $(AUTH_LATEST)
	python3 ../ci/jinja2_render.py '{"service_base_image":{"image":"service-base"}}' Dockerfile Dockerfile.out
	docker build -f Dockerfile.out -t auth --cache-from auth,$(AUTH_LATEST),base ..

.PHONY: push
push: build
	docker tag auth $(AUTH_LATEST)
	docker push $(AUTH_LATEST)
	docker tag auth $(AUTH_IMAGE)
	docker push $(AUTH_IMAGE)

.PHONY: deploy
deploy: push
	! [ -z $(NAMESPACE) ]  # call this like: make deploy NAMESPACE=default
	kubectl -n $(NAMESPACE) apply -f auth-driver-service-account.yaml
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":$(DEPLOY),"default_ns":{"name":"$(NAMESPACE)"},"auth_image":{"image":"$(AUTH_IMAGE)"},"auth_database":{"user_secret_name":"sql-auth-user-config"},"global":{"project":"$(PROJECT)","zone":"$(ZONE)","domain":"$(DOMAIN)"}}' deployment.yaml deployment.yaml.out
	kubectl -n $(NAMESPACE) apply -f deployment.yaml.out

.PHONY:
clean:
	rm -f Dockerfile.out deployment.yaml.out
