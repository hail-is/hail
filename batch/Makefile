.PHONY: build build-test \
  push push-test \
  run-docker run \
  test test-local deploy

PROJECT = $(shell gcloud config get-value project)

build:
	docker build -t batch .

build-test:
	docker build -t batch-test -f Dockerfile.test .

push: IMAGE="gcr.io/$(PROJECT)/batch:$(shell docker images -q --no-trunc batch | sed -e 's,[^:]*:,,')"
push: build
	docker tag batch $(IMAGE)
	docker push $(IMAGE)
	echo $(IMAGE) > batch-image

push-test: IMAGE="gcr.io/$(PROJECT)/batch-test:$(shell docker images -q --no-trunc batch-test | sed -e 's,[^:]*:,,')"
push-test: build-test
	docker tag batch-test $(IMAGE)
	docker push $(IMAGE)
	echo $(IMAGE) > batch-test-image

run-docker: build
	docker run -e BATCH_USE_KUBE_CONFIG=1 -i -v $(HOME)/.kube:/root/.kube -p 5000:5000 -t batch

run:
	BATCH_USE_KUBE_CONFIG=1 python -c 'import batch.server; batch.server.serve()'

test: push-test
	sed -e "s,@image@,$$(cat batch-test-image)," \
	  < test-batch-pod.yaml.in > test-batch-pod.yaml
	kubectl create -f test-batch-pod.yaml

test-local:
	POD_NAMESPACE='test' BATCH_USE_KUBE_CONFIG=1 ./test-locally.sh

deploy: push
	sed -e "s,@sha@,$$(git rev-parse --short=12 HEAD)," \
	  -e "s,@image@,$$(cat batch-image)," \
	  < deployment.yaml.in > deployment.yaml
	kubectl -n default apply -f deployment.yaml
