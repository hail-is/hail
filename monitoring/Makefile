include ../config.mk

MONITORING_LATEST = $(DOCKER_PREFIX)/monitoring:latest
MONITORING_IMAGE = $(DOCKER_PREFIX)/monitoring:$(shell docker images -q --no-trunc monitoring:latest | sed -e 's,[^:]*:,,')

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
	$(MAKE) -C ../docker build
	-docker pull $(MONITORING_LATEST)
	python3 ../ci/jinja2_render.py '{"service_base_image":{"image":"service-base"}}' Dockerfile Dockerfile.out
	docker build -f Dockerfile.out -t monitoring --cache-from monitoring,$(MONITORING_LATEST),service-base ..

.PHONY: push
push: build
	docker tag monitoring $(MONITORING_LATEST)
	docker push $(MONITORING_LATEST)
	docker tag monitoring $(MONITORING_IMAGE)
	docker push $(MONITORING_IMAGE)

.PHONY: deploy
deploy: push
	! [ -z $(NAMESPACE) ]  # call this like: make deploy NAMESPACE=default
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":$(DEPLOY),"default_ns":{"name":"$(NAMESPACE)"},"monitoring_image":{"image":"$(MONITORING_IMAGE)"},"monitoring_database":{"user_secret_name":"sql-monitoring-user-config"},"global":{"project":"$(PROJECT)","zone":"$(ZONE)","domain":"$(DOMAIN)"}}' deployment.yaml deployment.yaml.out
	kubectl -n $(NAMESPACE) apply -f deployment.yaml.out

.PHONY: clean
clean:
	rm -f Dockerfile.out deployment.yaml.out
