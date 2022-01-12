include ../config.mk

TOKEN = $(shell cat /dev/urandom | LC_ALL=C tr -dc 'a-z0-9' | head -c 12)

BATCH_IMAGE := $(DOCKER_PREFIX)/batch:$(TOKEN)
BATCH_WORKER_IMAGE := $(DOCKER_PREFIX)/batch-worker:$(TOKEN)

EXTRA_PYTHONPATH := ../hail/python:../gear:../web_common
PYTHON := PYTHONPATH=$${PYTHONPATH:+$${PYTHONPATH}:}$(EXTRA_PYTHONPATH) python3

BLACK := $(PYTHON) -m black . --line-length=120 --skip-string-normalization

.PHONY: check
check:
	$(PYTHON) -m flake8  --config ../setup.cfg batch
	$(PYTHON) -m pylint --rcfile ../pylintrc batch --score=n
	$(PYTHON) -m mypy --config-file ../setup.cfg batch
	$(BLACK) --check --diff
	curlylint .
	bash ../check-sql.sh

.PHONY: build-batch
build-batch:
	$(MAKE) -C ../docker service-base
	python3 ../ci/jinja2_render.py '{"service_base_image":{"image":"'$$(cat ../docker/service-base-image-ref)'"}}' Dockerfile Dockerfile.out
	../docker-build.sh . Dockerfile.out $(BATCH_IMAGE)

.PHONY: build-worker
build-worker:
	python3 ../ci/jinja2_render.py '{"hail_ubuntu_image":{"image":"'$$(cat ../docker/hail-ubuntu-image-ref)'"}}' Dockerfile.worker Dockerfile.worker.out
	../docker-build.sh .. batch/Dockerfile.worker.out $(BATCH_WORKER_IMAGE)

.PHONY: build
build: build-batch build-worker

JINJA_ENVIRONMENT = '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":$(DEPLOY),"batch_image":{"image":"$(BATCH_IMAGE)"},"batch_worker_image":{"image":"$(BATCH_WORKER_IMAGE)"},"default_ns":{"name":"$(NAMESPACE)"},"batch_database":{"user_secret_name":"sql-batch-user-config"},"scope":"$(SCOPE)"}'

.PHONY: deploy
deploy: build
	! [ -z $(NAMESPACE) ]  # call this like: make deploy NAMESPACE=default
	E=$(JINJA_ENVIRONMENT) && \
	  python3 ../ci/jinja2_render.py $$E deployment.yaml deployment.yaml.out && \
	  python3 ../ci/jinja2_render.py $$E service-account.yaml service-account.yaml.out
	kubectl -n $(NAMESPACE) apply -f service-account.yaml.out
	kubectl -n $(NAMESPACE) apply -f deployment.yaml.out

.PHONY: gcp-create-build-worker-image-instance
gcp-create-build-worker-image-instance:
	-gcloud -q compute --project $(PROJECT) instances delete --zone=$(ZONE) build-batch-worker-image
	python3 ../ci/jinja2_render.py '{"global":{"docker_root_image":"$(DOCKER_ROOT_IMAGE)"}}' build-batch-worker-image-startup-gcp.sh build-batch-worker-image-startup-gcp.sh.out
	gcloud -q compute --project $(PROJECT) instances create --zone=$(ZONE) build-batch-worker-image --machine-type=n1-standard-1 --network=default --network-tier=PREMIUM --metadata-from-file startup-script=build-batch-worker-image-startup-gcp.sh.out --no-restart-on-failure --maintenance-policy=MIGRATE --scopes=https://www.googleapis.com/auth/cloud-platform --image=$$(gcloud compute images list --standard-images --filter 'family="ubuntu-minimal-2004-lts"' --format='value(name)') --image-project=ubuntu-os-cloud --boot-disk-size=10GB --boot-disk-type=pd-ssd

.PHONY: gcp-create-worker-image
gcp-create-worker-image:
	-gcloud -q compute --project $(PROJECT) images delete batch-worker-12
	gcloud -q compute --project $(PROJECT) images create batch-worker-12 --source-disk=build-batch-worker-image --source-disk-zone=$(ZONE)
	gcloud -q compute --project $(PROJECT) instances delete --zone=$(ZONE) build-batch-worker-image
