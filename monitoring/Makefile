include ../config.mk

TOKEN = $(shell cat /dev/urandom | LC_ALL=C tr -dc 'a-z0-9' | head -c 12)

MONITORING_IMAGE := $(DOCKER_PREFIX)/monitoring:$(TOKEN)

PYTHONPATH := $${PYTHONPATH:+$${PYTHONPATH}:}
PYTHON := PYTHONPATH=$(PYTHONPATH)../hail/python:../gear:../web_common python3

.PHONY: check
check:
	$(PYTHON) -m flake8 monitoring
	$(PYTHON) -m pylint --rcfile ../pylintrc monitoring --score=n
	curlylint .
	bash ../check-sql.sh

.PHONY: build
build:
	$(MAKE) -C ../docker service-base
	python3 ../ci/jinja2_render.py '{"service_base_image":{"image":"'$$(cat ../docker/service-base-image-ref)'"}}' Dockerfile Dockerfile.out
	../docker-build.sh .. monitoring/Dockerfile.out $(MONITORING_IMAGE)

.PHONY: deploy
deploy: build
	! [ -z $(NAMESPACE) ]  # call this like: make deploy NAMESPACE=default
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":$(DEPLOY),"default_ns":{"name":"$(NAMESPACE)"},"monitoring_image":{"image":"$(MONITORING_IMAGE)"},"monitoring_database":{"user_secret_name":"sql-monitoring-user-config"},"global":{"project":"$(PROJECT)","zone":"$(ZONE)","domain":"$(DOMAIN)"}}' deployment.yaml deployment.yaml.out
	kubectl -n $(NAMESPACE) apply -f deployment.yaml.out

.PHONY: clean
clean:
	rm -f Dockerfile.out deployment.yaml.out
