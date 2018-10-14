.PHONY: hail-ci-build-image push-hail-ci-build-image

PROJECT = $(shell gcloud config get-value project)

hail-ci-build-image:
	docker build -t batch-pr-builder -f Dockerfile.pr-builder .
	echo "gcr.io/$(PROJECT)/batch-pr-builder:`docker images -q --no-trunc batch-pr-builder | sed -e 's,[^:]*:,,'`" > ../hail-ci-build-image
	docker tag batch-pr-builder `cat ../hail-ci-build-image`

push-hail-ci-build-image: hail-ci-build-image
	docker push `cat ../hail-ci-build-image`

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
	echo $(IMAGE) > batch-test-image
	docker tag batch $(IMAGE)
	docker push $(IMAGE)

run-docker:
	docker run -e BATCH_USE_KUBE_CONFIG=1 -i -v $(HOME)/.kube:/root/.kube -p 5000:5000 -t batch

run:
	BATCH_USE_KUBE_CONFIG=1 python batch/server.py

test-local:
	POD_IP='127.0.0.1' BATCH_URL='http://127.0.0.1:5000' python -m unittest -v test/test_batch.py

deploy-batch:
	sed -e "s,@sha@,$$(git rev-parse --short=12 HEAD)," \
	  -e "s,@image@,$$(cat batch-image)," \
	  < deployment.yaml.in > deployment.yaml
	kubectl apply -f deployment.yaml
# ci can't recover from batch restart yet
	kubectl rollout status deployment batch-deployment
	kubectl delete pods -l app=hail-ci
