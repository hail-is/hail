.PHONY: build build-batch build-batch-test \
  push push-batch push-batch-test \
  run-docker run \
  test test-local deploy-batch

PROJECT = $(shell gcloud config get-value project)

build: build-batch build-batch-test

build-batch:
	docker build -t batch .

build-batch-test:
	docker build -t batch-test -f Dockerfile.test .

push: push-batch push-batch-test

push-batch: IMAGE="gcr.io/$(PROJECT)/batch:$(shell docker images -q --no-trunc batch | sed -e 's,[^:]*:,,')"
push-batch: build-batch
	docker tag batch $(IMAGE)
	docker push $(IMAGE)
	echo $(IMAGE) > batch-image

push-batch-test: IMAGE="gcr.io/$(PROJECT)/batch-test:$(shell docker images -q --no-trunc batch-test | sed -e 's,[^:]*:,,')"
push-batch-test: build-batch-test
	docker tag batch-test $(IMAGE)
	docker push $(IMAGE)
	echo $(IMAGE) > batch-test-image

run-docker:
	docker run -e BATCH_USE_KUBE_CONFIG=1 -i -v $(HOME)/.kube:/root/.kube -p 5000:5000 -t batch

run:
	BATCH_USE_KUBE_CONFIG=1 python batch/server.py

test:
	sed -e "s,@image@,$$(cat batch-test-image)," \
	  < test-batch-pod.yaml.in > test-batch-pod.yaml
	kubectl create -f test-batch-pod.yaml

test-local:
	POD_IP='127.0.0.1' BATCH_URL='http://127.0.0.1:5000' python -m unittest -v test/test_batch.py

deploy-batch: push-batch
	sed -e "s,@sha@,$$(git rev-parse --short=12 HEAD)," \
	  -e "s,@image@,$$(cat batch-image)," \
	  < deployment.yaml.in > deployment.yaml
	kubectl -n default apply -f deployment.yaml
