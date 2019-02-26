.PHONY: build build-test \
  push push-test \
  run-docker run \
  test test-local deploy clean

PROJECT = $(shell gcloud config get-value project)
PY_FILES = $(shell find batch -iname \*.py -not -exec git check-ignore -q {} \; -print)
PY_CHECKERS = build/flake8 build/pylint
BATCH_PORT ?= 5000

clean:
	-rm -r build

build/conda-env: environment.yml
	conda env update -f environment.yml || conda env create -f environment.yml
	mkdir -p build && touch $@

build/flake8: $(PY_FILES) build/conda-env
	. ../loadconda && conda activate hail-batch && python -m flake8 batch
	mkdir -p build && touch $@

build/pylint: $(PY_FILES) build/conda-env
	. ../loadconda && conda activate hail-batch && python -m pylint --rcfile batch/pylintrc batch --score=n
	mkdir -p build && touch $@

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

run: build/conda-env
	. ../loadconda && conda activate hail-batch && python -c 'import batch.server; batch.server.serve('$(BATCH_PORT)')'

run-local: run
	. ../loadconda && conda activate hail-batch && BATCH_USE_KUBE_CONFIG=1 python -c 'import batch.server; batch.server.serve('$(BATCH_PORT)')'

test: push-test
	sed -e "s,@image@,$$(cat batch-test-image)," \
	  < test-batch-pod.yaml.in > test-batch-pod.yaml
	kubectl create -f test-batch-pod.yaml

ifneq ($(PY_CHECK),false)
test-local: $(PY_CHECKERS)
endif
test-local: build/conda-env
	. ../loadconda && conda activate hail-batch && POD_NAMESPACE='test' BATCH_USE_KUBE_CONFIG=1 ./test-locally.sh

# local means server and test client are two processes on one machine
# in-cluster means in a k8s pod (from which we get k8s creds)
test-local-in-cluster: build/conda-env $(PY_CHECKERS)
	. ../loadconda && conda activate hail-batch && POD_NAMESPACE='test' ./test-locally.sh

deploy: push
	sed -e "s,@sha@,$$(git rev-parse --short=12 HEAD)," \
	  -e "s,@image@,$$(cat batch-image)," \
	  < deployment.yaml.in > deployment.yaml
	kubectl -n default apply -f deployment.yaml
