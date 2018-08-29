.PHONY: hail-ci-build-image

hail-ci-build-image:
	docker build -t batch-pr-builder -f Dockerfile.pr-builder .
	echo "gcr.io/broad-ctsa/batch-pr-builder:`docker images -q --no-trunc batch-pr-builder | sed -e 's,[^:]*:,,'`" > hail-ci-build-image
	docker tag batch-pr-builder `cat hail-ci-build-image`

push-hail-ci-build-image: hail-ci-build-image
	docker push `cat hail-ci-build-image`

build: build-batch build-test

build-batch:
	docker build -t gcr.io/broad-ctsa/batch -t batch .

build-test:
	docker build -t gcr.io/broad-ctsa/batch-test -t batch-test -f Dockerfile.test .

push: push-batch push-test

push-batch:
	docker push gcr.io/broad-ctsa/batch

push-test:
	docker push gcr.io/broad-ctsa/batch-test

redeploy: build push
	kubectl delete -f deployment.yaml
	sleep 5
	kubectl create -f deployment.yaml

run-docker:
	docker run -e BATCH_USE_KUBE_CONFIG=1 -i -v $(HOME)/.kube:/root/.kube -p 5000:5000 -t batch

run:
	BATCH_USE_KUBE_CONFIG=1 python batch/server.py

test-local:
	POD_IP='127.0.0.1' BATCH_URL='http://127.0.0.1:5000' python -m unittest -v test/test_batch.py
