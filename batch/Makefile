include ../config.mk

TOKEN = $(shell cat /dev/urandom | LC_ALL=C tr -dc 'a-z0-9' | head -c 12)

BATCH_IMAGE := $(DOCKER_PREFIX)/batch:$(TOKEN)
BATCH_WORKER_IMAGE := $(DOCKER_PREFIX)/batch-worker:$(TOKEN)

BATCH_DRIVER_NGINX_IMAGE := $(DOCKER_PREFIX)/batch-driver-nginx:$(TOKEN)

EXTRA_PYTHONPATH := ../hail/python:../gear:../web_common
PYTHON := PYTHONPATH=$${PYTHONPATH:+$${PYTHONPATH}:}$(EXTRA_PYTHONPATH) python3
CLOUD := $(shell kubectl get secret global-config --template={{.data.cloud}} | base64 --decode)

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

.PHONY: build-batch
build-batch:
	$(MAKE) -C ../docker service-base
	python3 ../ci/jinja2_render.py '{"service_base_image":{"image":"'$$(cat ../docker/service-base-image-ref)'"}}' Dockerfile Dockerfile.out
	../docker-build.sh . Dockerfile.out $(BATCH_IMAGE)

jars/junixsocket-selftest-2.3.3-jar-with-dependencies.jar:
	mkdir jars
	cd jars && curl -LO https://github.com/kohlschutter/junixsocket/releases/download/junixsocket-parent-2.3.3/junixsocket-selftest-2.3.3-jar-with-dependencies.jar

src/main/java/is/hail/JVMEntryway.class: src/main/java/is/hail/JVMEntryway.java jars/junixsocket-selftest-2.3.3-jar-with-dependencies.jar
	javac -cp jars/junixsocket-selftest-2.3.3-jar-with-dependencies.jar $<

.PHONY: build-batch-driver-nginx
build-batch-driver-nginx:
	python3 ../ci/jinja2_render.py '{"hail_ubuntu_image":{"image":"'$$(cat ../docker/hail-ubuntu-image-ref)'"}}' Dockerfile.driver-nginx Dockerfile.driver-nginx.out
	../docker-build.sh . Dockerfile.driver-nginx.out $(BATCH_DRIVER_NGINX_IMAGE)

.PHONY: build-worker
build-worker: src/main/java/is/hail/JVMEntryway.class
	python3 ../ci/jinja2_render.py '{"hail_ubuntu_image":{"image":"'$$(cat ../docker/hail-ubuntu-image-ref)'"},"global":{"cloud":"$(CLOUD)"}}' Dockerfile.worker Dockerfile.worker.out
	../docker-build.sh .. batch/Dockerfile.worker.out $(BATCH_WORKER_IMAGE)

.PHONY: build
build: build-batch build-batch-driver-nginx build-worker

JINJA_ENVIRONMENT = '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":$(DEPLOY),"batch_image":{"image":"$(BATCH_IMAGE)"},"batch_worker_image":{"image":"$(BATCH_WORKER_IMAGE)"},"default_ns":{"name":"$(NAMESPACE)"},"batch_database":{"user_secret_name":"sql-batch-user-config"},"scope":"$(SCOPE)","batch_driver_nginx_image":{"image":"$(BATCH_DRIVER_NGINX_IMAGE)"}}'

.PHONY: deploy
deploy: build
	! [ -z $(NAMESPACE) ]  # call this like: make deploy NAMESPACE=default
	E=$(JINJA_ENVIRONMENT) && \
	  python3 ../ci/jinja2_render.py $$E deployment.yaml deployment.yaml.out && \
	  python3 ../ci/jinja2_render.py $$E service-account.yaml service-account.yaml.out
	kubectl -n $(NAMESPACE) apply -f service-account.yaml.out
	kubectl -n $(NAMESPACE) apply -f deployment.yaml.out
