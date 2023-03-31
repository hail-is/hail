include ../config.mk

EXTRA_PYTHONPATH := ../hail/python:../gear:../web_common
PYTHON := PYTHONPATH=$${PYTHONPATH:+$${PYTHONPATH}:}$(EXTRA_PYTHONPATH) python3

BLACK := $(PYTHON) -m black . --line-length=120 --skip-string-normalization

.PHONY: check
check:
	$(PYTHON) -m flake8  --config ../setup.cfg batch
	$(PYTHON) -m pylint --rcfile ../pylintrc batch --score=n
	$(PYTHON) -m mypy --config-file ../setup.cfg batch
	$(PYTHON) -m isort . --check-only --diff
	$(BLACK) --check --diff
	curlylint .
	bash ../check-sql.sh

.PHONY: build
build:
	$(MAKE) -C .. batch-image batch-worker-image

JINJA_ENVIRONMENT = '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":$(DEPLOY),"batch_image":{"image":"$(shell cat ../batch-image)"},"batch_worker_image":{"image":"$(shell cat ../batch-worker-image)"},"default_ns":{"name":"$(NAMESPACE)"},"batch_database":{"user_secret_name":"sql-batch-user-config"},"scope":"$(SCOPE)","global":{"docker_prefix":"$(DOCKER_PREFIX)"}}'

.PHONY: deploy
deploy: build
	! [ -z $(NAMESPACE) ]  # call this like: make deploy NAMESPACE=default
	E=$(JINJA_ENVIRONMENT) && \
	  python3 ../ci/jinja2_render.py $$E deployment.yaml deployment.yaml.out && \
	  python3 ../ci/jinja2_render.py $$E service-account.yaml service-account.yaml.out
	kubectl -n $(NAMESPACE) apply -f service-account.yaml.out
	kubectl -n $(NAMESPACE) apply -f deployment.yaml.out
