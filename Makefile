build-utils:
	docker build -t true -t gcr.io/broad-ctsa/true -f utils/Dockerfile.true utils
	docker build -t false -t gcr.io/broad-ctsa/false -f utils/Dockerfile.false utils
	docker build -t echo -t gcr.io/broad-ctsa/echo -f utils/Dockerfile.echo utils

push-utils:
	docker push gcr.io/broad-ctsa/true
	docker push gcr.io/broad-ctsa/false
	docker push gcr.io/broad-ctsa/echo

build:
	docker build -t gcr.io/broad-ctsa/batch -t batch .

build-test:
	docker build -t gcr.io/broad-ctsa/batch-test -t batch-test -f Dockerfile.test .

push:
	docker push gcr.io/broad-ctsa/batch

push-test:
	docker push gcr.io/broad-ctsa/batch-test

run-docker:
	docker run -e BATCH_USE_KUBE_CONFIG=1 -i -v $(HOME)/.kube:/root/.kube -p 5000:5000 -t batch

run:
	BATCH_USE_KUBE_CONFIG=1 python batch.py

test-local:
	BATCH_URL='http://localhost:5000' python -m unittest test/test_batch.py
