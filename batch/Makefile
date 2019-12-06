PROJECT := $(shell gcloud config get-value project)
ZONE := us-central1-a
KUBERNETES_SERVER_URL := https://104.198.230.143

BATCH_LATEST = gcr.io/$(PROJECT)/batch:latest
BATCH_IMAGE = gcr.io/$(PROJECT)/batch:$(shell docker images -q --no-trunc batch | sed -e 's,[^:]*:,,')

BATCH_WORKER_LATEST = gcr.io/$(PROJECT)/batch-worker:latest
BATCH_WORKER_IMAGE = gcr.io/$(PROJECT)/batch-worker:$(shell docker images -q --no-trunc batch-worker | sed -e 's,[^:]*:,,')

PYTHONPATH := $${PYTHONPATH:+$${PYTHONPATH}:}../hail/python:../gear:../web_common
PYTHON := PYTHONPATH=$(PYTHONPATH) python3

check:
	$(PYTHON) -m flake8  --config ../setup.cfg batch
	$(PYTHON) -m pylint --rcfile ../pylintrc batch --score=n

build:
	make -C ../docker build
	-docker pull $(BATCH_LATEST)
	python3 ../ci/jinja2_render.py '{"service_base_image":{"image":"service-base"}}' Dockerfile Dockerfile.out
	docker build -t batch -f Dockerfile.out --cache-from batch,$(BATCH_LATEST),service-base ..
	-docker pull python:3.6-slim-stretch
	-docker pull $(BATCH_WORKER_LATEST)
	docker build -t batch-worker -f Dockerfile.worker --cache-from batch-worker,$(BATCH_WORKER_LATEST),python:3.6-slim-stretch ..

push: build
	docker tag batch $(BATCH_LATEST)
	docker push $(BATCH_LATEST)
	docker tag batch $(BATCH_IMAGE)
	docker push $(BATCH_IMAGE)
	docker tag batch-worker $(BATCH_WORKER_LATEST)
	docker push $(BATCH_WORKER_LATEST)
	docker tag batch-worker $(BATCH_WORKER_IMAGE)
	docker push $(BATCH_WORKER_IMAGE)

deploy: push
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":true,"batch_image":{"image":"$(BATCH_IMAGE)"},"batch_worker_image":{"image":"$(BATCH_WORKER_IMAGE)"},"default_ns":{"name":"default"},"batch_pods_ns":{"name":"batch-pods"},"batch_database":{"user_secret_name":"sql-batch-batch-admin-config"},"global":{"project":"$(PROJECT)","zone":"$(ZONE)","domain":"$(DOMAIN)","k8s_server_url":"$(KUBERNETES_SERVER_URL)"}}' deployment.yaml deployment.yaml.out
	kubectl -n default apply -f deployment.yaml.out

create-build-worker-image-instance:
	-gcloud -q compute --project $(PROJECT) instances delete --zone=$(ZONE) build-batch-worker-image
	gcloud -q compute --project $(PROJECT) instances create --zone=$(ZONE) build-batch-worker-image --machine-type=n1-standard-1 --network=default --network-tier=PREMIUM --metadata-from-file startup-script=build-batch-worker-image-startup.sh --no-restart-on-failure --maintenance-policy=MIGRATE --scopes=https://www.googleapis.com/auth/cloud-platform --image=ubuntu-minimal-1804-bionic-v20191024 --image-project=ubuntu-os-cloud --boot-disk-size=10GB --boot-disk-type=pd-ssd

create-worker-image:
	gcloud -q compute --project $(PROJECT) images delete batch-worker-7
	gcloud -q compute --project $(PROJECT) images create batch-worker-7 --source-disk=build-batch-worker-image --source-disk-zone=$(ZONE)
	gcloud -q compute --project $(PROJECT) instances delete --zone=$(ZONE) build-batch-worker-image
