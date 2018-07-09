build-utils:
	docker build -t true -t gcr.io/broad-ctsa/true -f utils/Dockerfile.true utils
	docker build -t false -t gcr.io/broad-ctsa/false -f utils/Dockerfile.false utils
	docker build -t echo -t gcr.io/broad-ctsa/echo -f utils/Dockerfile.echo utils

push-utils:
	docker push gcr.io/broad-ctsa/true
	docker push gcr.io/broad-ctsa/false
	docker push gcr.io/broad-ctsa/echo

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

run-docker:
	docker run -e BATCH_USE_KUBE_CONFIG=1 -i -v $(HOME)/.kube:/root/.kube -p 5000:5000 -t batch

run:
	BATCH_USE_KUBE_CONFIG=1 python batch/server.py

test-local:
	POD_IP='127.0.0.1' BATCH_URL='http://127.0.0.1:5000' python -m unittest test/test_batch.py
