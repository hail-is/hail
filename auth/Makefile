PROJECT = $(shell gcloud config get-value project)
DOMAIN ?= hail.is

AUTH_LATEST = gcr.io/$(PROJECT)/auth:latest
AUTH_IMAGE = gcr.io/$(PROJECT)/auth:$(shell docker images -q --no-trunc auth:latest | sed -e 's,[^:]*:,,')

PYTHONPATH := $${PYTHONPATH:+$${PYTHONPATH}:}
PYTHON := PYTHONPATH=$(PYTHONPATH)../hail/python:../gear:../web_common python3

.PHONY: check
check:
	$(PYTHON) -m flake8 auth
	$(PYTHON) -m pylint --rcfile ../pylintrc auth --score=n

.PHONY: build
build:
	make -C ../docker build
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
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":true,"default_ns":{"name":"default"},"batch_pods_ns":{"name":"batch-pods"},"auth_image":{"image":"$(AUTH_IMAGE)"},"auth_database":{"user_secret_name":"sql-auth-user-config"},"global":{"project":"$(PROJECT)","zone":"$(ZONE)","domain":"$(DOMAIN)"}}' deployment.yaml deployment.yaml.out
	kubectl -n default apply -f deployment.yaml.out

.PHONY:
clean:
	rm -f Dockerfile.out deployment.yaml.out
