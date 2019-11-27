PROJECT := $(shell gcloud config get-value project)
ZONE := us-central1-a
KUBERNETES_SERVER_URL := https://104.198.230.143

BATCH2_LATEST = gcr.io/$(PROJECT)/batch2:latest
BATCH2_IMAGE = gcr.io/$(PROJECT)/batch2:$(shell docker images -q --no-trunc batch2 | sed -e 's,[^:]*:,,')

BATCH2_WORKER_LATEST = gcr.io/$(PROJECT)/batch2-worker:latest
BATCH2_WORKER_IMAGE = gcr.io/$(PROJECT)/batch2-worker:$(shell docker images -q --no-trunc batch2-worker | sed -e 's,[^:]*:,,')

PYTHONPATH := $${PYTHONPATH:+$${PYTHONPATH}:}../hail/python:../gear:../web_common
PYTHON := PYTHONPATH=$(PYTHONPATH) python3

check:
	$(PYTHON) -m flake8  --config ../setup.cfg batch
	$(PYTHON) -m pylint --rcfile ../pylintrc batch --score=n

build:
	make -C ../docker build
	-docker pull $(BATCH2_LATEST)
	python3 ../ci/jinja2_render.py '{"service_base_image":{"image":"service-base"}}' Dockerfile Dockerfile.out
	docker build -t batch2 -f Dockerfile.out --cache-from batch2,$(BATCH2_LATEST),service-base ..
	-docker pull python:3.6-slim-stretch
	-docker pull $(BATCH2_WORKER_LATEST)
	docker build -t batch2-worker -f Dockerfile.worker --cache-from batch2-worker,$(BATCH2_WORKER_LATEST),python:3.6-slim-stretch ..

push: build
	docker tag batch2 $(BATCH2_LATEST)
	docker push $(BATCH2_LATEST)
	docker tag batch2 $(BATCH2_IMAGE)
	docker push $(BATCH2_IMAGE)
	docker tag batch2 $(BATCH2_WORKER_LATEST)
	docker push $(BATCH2_WORKER_LATEST)
	docker tag batch2 $(BATCH2_WORKER_IMAGE)
	docker push $(BATCH2_WORKER_IMAGE)

deploy: push
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":true,"batch2_image":{"image":"$(BATCH2_IMAGE)"},"batch2_worker_image":{"image":"$(BATCH2_WORKER_IMAGE)"},"default_ns":{"name":"default"},"batch_pods_ns":{"name":"batch-pods"},"batch2_database":{"user_secret_name":"sql-batch2-batch2-admin-config"},"global":{"project":"$(PROJECT)","zone":"$(ZONE)","domain":"$(DOMAIN)","k8s_server_url":"$(KUBERNETES_SERVER_URL)"}}' deployment.yaml deployment.yaml.out
	kubectl -n default apply -f deployment.yaml.out

create-build-worker-image-instance:
	-gcloud -q compute --project $(PROJECT) instances delete --zone=$(ZONE) build-batch2-worker-image
	gcloud -q compute --project $(PROJECT) instances create --zone=$(ZONE) build-batch2-worker-image --machine-type=n1-standard-1 --network=default --network-tier=PREMIUM --metadata-from-file startup-script=build-batch2-worker-image-startup.sh --no-restart-on-failure --maintenance-policy=MIGRATE --scopes=https://www.googleapis.com/auth/cloud-platform --image=ubuntu-minimal-1804-bionic-v20191024 --image-project=ubuntu-os-cloud --boot-disk-size=10GB --boot-disk-type=pd-ssd

create-worker-image:
	gcloud -q compute --project $(PROJECT) images delete batch2-worker-6
	gcloud -q compute --project $(PROJECT) images create batch2-worker-6 --source-disk=build-batch2-worker-image --source-disk-zone=$(ZONE)
	gcloud -q compute --project $(PROJECT) instances delete --zone=$(ZONE) build-batch2-worker-image
