.PHONY: build build-test \
  push push-test \
  run-docker run \
  test test-local deploy test-deploy \
  check clean \
  install-cloud-sql-proxy local-cloud-sql-config run-cloud-sql-proxy

PROJECT = $(shell gcloud config get-value project)
BATCH_LATEST = gcr.io/$(PROJECT)/batch:latest
BATCH_IMAGE = gcr.io/$(PROJECT)/batch:$(shell docker images -q --no-trunc batch | sed -e 's,[^:]*:,,')

BATCH_TEST_LATEST = gcr.io/$(PROJECT)/batch-test:latest
BATCH_TEST_IMAGE = gcr.io/$(PROJECT)/batch-test:$(shell docker images -q --no-trunc batch-test | sed -e 's,[^:]*:,,')

PY_FILES = $(shell find batch -iname \*.py -not -exec git check-ignore -q {} \; -print)
PY_CHECKERS = flake8-stmp pylint-stmp

BATCH_PORT ?= 5000
CLOUD_SQL_PORT ?= 3307

flake8-stmp: $(PY_FILES)
	python3 -m flake8 batch
	touch $@

pylint-stmp: $(PY_FILES)
	python3 -m pylint --rcfile batch/pylintrc batch --score=n
	touch $@

check: $(PY_CHECKERS)

build:
	make -C ../docker build
	-docker pull $(BATCH_LATEST)
	docker build -t batch --cache-from batch,$(BATCH_LATEST),base,ubuntu:18.04 .

build-test:
	docker build -t batch-test -f Dockerfile.test .

# https://cloud.google.com/sql/docs/mysql/quickstart-proxy-test
install-cloud-sql-proxy:
	test -f cloud_sql_proxy || \
		curl -o cloud_sql_proxy https://dl.google.com/cloudsql/cloud_sql_proxy.darwin.amd64 && \
		chmod +x cloud_sql_proxy

run-cloud-sql-proxy: local-cloud-sql-config install-cloud-sql-proxy
	$(eval config := $(shell pwd)/secrets/batch-test-cloud-sql-config.json)
	$(eval connection_name := $(shell jq -r '.connection_name' $(config)))
	$(eval port := $(shell jq -r '.port' $(config)))
	./cloud_sql_proxy -instances=$(connection_name)=tcp:$(port) &

local-cloud-sql-config:
	mkdir -p secrets
	kubectl get secret batch-test-cloud-sql-config -n batch-pods -o "jsonpath={.data.batch-test-cloud-sql-config\.json}" | base64 --decode > secrets/batch-test-cloud-sql-config.json
	jq '.host = "127.0.0.1" | .port = $(CLOUD_SQL_PORT)' secrets/batch-test-cloud-sql-config.json > secrets/tmp.batch-test-cloud-sql-config.json
	mv secrets/tmp.batch-test-cloud-sql-config.json secrets/batch-test-cloud-sql-config.json

push: build
	docker tag batch $(BATCH_LATEST)
	docker push $(BATCH_LATEST)
	docker tag batch $(BATCH_IMAGE)
	docker push $(BATCH_IMAGE)

push-test: build-test
	docker tag batch-test $(BATCH_TEST_LATEST)
	docker push $(BATCH_TEST_LATEST)
	docker tag batch-test $(BATCH_TEST_IMAGE)
	docker push $(BATCH_TEST_IMAGE)

run-docker: build
	docker run -e BATCH_USE_KUBE_CONFIG=1 -i -v $(HOME)/.kube:/root/.kube -p 5000:5000 -t batch

run:
	python3 -c 'import batch.server; batch.server.serve('$(BATCH_PORT)')'

run-local: run-cloud-sql-proxy
	CLOUD_SQL_CONFIG_PATH=secrets/batch-test-cloud-sql-config.json \
            BATCH_USE_KUBE_CONFIG=1 \
            python3 -c 'import batch.server; batch.server.serve('$(BATCH_PORT)')'

test: push-test
	sed -e "s,@image@,$(BATCH_TEST_IMAGE)," \
	  < test-batch-pod.yaml.in > test-batch-pod.yaml
	kubectl create -f test-batch-pod.yaml

ifneq ($(PY_CHECK),false)
test-local: check
endif
test-local: install-cloud-sql-proxy local-cloud-sql-config
	POD_NAMESPACE='test' BATCH_USE_KUBE_CONFIG=1 CLOUD_SQL_PROXY=1 ./test-locally.sh

# local means server and test client are two processes on one machine
# in-cluster means in a k8s pod (from which we get k8s creds)
test-local-in-cluster: check
	POD_NAMESPACE='test' ./test-locally.sh

deploy: push
	sed -e "s,@sha@,$$(git rev-parse --short=12 HEAD)," \
	  -e "s,@image@,$(BATCH_IMAGE)," \
	  < deployment.yaml.in > deployment.yaml
	kubectl delete persistentvolumeclaim --all --namespace test
	kubectl delete persistentvolumeclaim --all --namespace batch-pods
	kubectl -n default apply -f deployment.yaml

test-deploy: push-test
	ifndef TEST_NAMESPACE
	$(error TEST_NAMESPACE is not set)
	endif
	sed -e "s,@sha@,$$(git rev-parse --short=12 HEAD)," \
	  -e "s,@image@,$(BATCH_TEST_IMAGE)," \
	  < deployment.yaml.in > deployment.yaml
	kubectl -n $(TEST_NAMESPACE) apply -f deployment.yaml

clean:
	rm -f $(PY_CHECKERS) && rm -rf secrets
