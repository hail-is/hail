include ../config.mk

BATCH_LATEST = $(DOCKER_PREFIX)/batch:latest
BATCH_IMAGE = $(DOCKER_PREFIX)/batch:$(shell docker images -q --no-trunc batch | sed -e 's,[^:]*:,,')

BATCH_WORKER_LATEST = $(DOCKER_PREFIX)/batch-worker:latest
BATCH_WORKER_IMAGE = $(DOCKER_PREFIX)/batch-worker:$(shell docker images -q --no-trunc batch-worker | sed -e 's,[^:]*:,,')

EXTRA_PYTHONPATH := ../hail/python:../gear:../web_common
PYTHON := PYTHONPATH=$${PYTHONPATH:+$${PYTHONPATH}:}$(EXTRA_PYTHONPATH) python3

.PHONY: check
check:
	$(PYTHON) -m flake8  --config ../setup.cfg batch
	$(PYTHON) -m pylint --rcfile ../pylintrc batch --score=n
	../check-sql.sh

.PHONY: build-prereqs
build-prereqs:
	$(MAKE) -C ../docker build

.PHONY: build-batch
build-batch: build-prereqs
	-docker pull $(BATCH_LATEST)
	python3 ../ci/jinja2_render.py '{"service_base_image":{"image":"service-base"}}' Dockerfile Dockerfile.out
	docker build -t batch -f Dockerfile.out --cache-from batch,$(BATCH_LATEST),service-base .

.PHONY: build-worker
build-worker: build-prereqs
	$(MAKE) -C ../hail shadowJar
# janky
	cp ../hail/build/libs/hail-all-spark.jar ./hail.jar
	-docker pull $(BATCH_WORKER_LATEST)
	python3 ../ci/jinja2_render.py '{"global":{"docker_prefix":"$(DOCKER_PREFIX)"}}}' Dockerfile.worker Dockerfile.worker.out
	docker build -t batch-worker -f Dockerfile.worker.out --cache-from batch-worker,$(BATCH_WORKER_LATEST),service-base ..

.PHONY: build
build: build-batch build-worker

.PHONY: push
push: build
	docker tag batch $(BATCH_LATEST)
	docker push $(BATCH_LATEST)
	docker tag batch $(BATCH_IMAGE)
	docker push $(BATCH_IMAGE)
	docker tag batch-worker $(BATCH_WORKER_LATEST)
	docker push $(BATCH_WORKER_LATEST)
	docker tag batch-worker $(BATCH_WORKER_IMAGE)
	docker push $(BATCH_WORKER_IMAGE)

JINJA_ENVIRONMENT = '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":$(DEPLOY),"batch_image":{"image":"$(BATCH_IMAGE)"},"batch_worker_image":{"image":"$(BATCH_WORKER_IMAGE)"},"default_ns":{"name":"$(NAMESPACE)"},"batch_database":{"user_secret_name":"sql-batch-user-config"},"global":{"project":"$(PROJECT)","domain":"$(DOMAIN)","k8s_server_url":"$(KUBERNETES_SERVER_URL)","docker_prefix":"$(DOCKER_PREFIX)"},"scope":"$(SCOPE)"}'

.PHONY: deploy
deploy: push
	! [ -z $(NAMESPACE) ]  # call this like: make deploy NAMESPACE=default
	E=$(JINJA_ENVIRONMENT) && \
	  python3 ../ci/jinja2_render.py $$E deployment.yaml deployment.yaml.out && \
	  python3 ../ci/jinja2_render.py $$E service-account.yaml service-account.yaml.out
	kubectl -n $(NAMESPACE) apply -f service-account.yaml.out
	kubectl -n $(NAMESPACE) apply -f deployment.yaml.out

.PHONY: create-build-worker-image-instance
create-build-worker-image-instance:
	-gcloud -q compute --project $(PROJECT) instances delete --zone=$(ZONE) build-batch-worker-image
	gcloud -q compute --project $(PROJECT) instances create --zone=$(ZONE) build-batch-worker-image --machine-type=n1-standard-1 --network=default --network-tier=PREMIUM --metadata-from-file startup-script=build-batch-worker-image-startup.sh --no-restart-on-failure --maintenance-policy=MIGRATE --scopes=https://www.googleapis.com/auth/cloud-platform --image=$$(gcloud compute images list --standard-images --filter 'family="ubuntu-minimal-2004-lts"' --format='value(name)') --image-project=ubuntu-os-cloud --boot-disk-size=10GB --boot-disk-type=pd-ssd

.PHONY: create-worker-image
create-worker-image:
	-gcloud -q compute --project $(PROJECT) images delete batch-worker-12
	gcloud -q compute --project $(PROJECT) images create batch-worker-12 --source-disk=build-batch-worker-image --source-disk-zone=$(ZONE)
	gcloud -q compute --project $(PROJECT) instances delete --zone=$(ZONE) build-batch-worker-image
