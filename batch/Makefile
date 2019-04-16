.PHONY: build build-test \
  push push-test \
  run-docker run \
  test test-local deploy test-deploy \
  check clean install-hailjwt

PROJECT := $(shell gcloud config get-value project)
BATCH_LATEST = gcr.io/$(PROJECT)/batch:latest
BATCH_IMAGE = gcr.io/$(PROJECT)/batch:$(shell docker images -q --no-trunc batch | sed -e 's,[^:]*:,,')

BATCH_TEST_LATEST = gcr.io/$(PROJECT)/batch-test:latest
BATCH_TEST_IMAGE = gcr.io/$(PROJECT)/batch-test:$(shell docker images -q --no-trunc batch-test | sed -e 's,[^:]*:,,')

PY_FILES = $(shell find batch -iname \*.py -not -exec git check-ignore -q {} \; -print)
PY_CHECKERS = flake8-stmp pylint-stmp

BATCH_PORT ?= 5000

include ../cloud-sql.mk

test/jwt-test-user-token: test/jwt-test-user.json test/jwt-test-secret-key install-hailjwt create_key.py
	python create_key.py test/jwt-test-secret-key < $< > $@

hailjwt: ../hailjwt/setup.py $(find ../hailjwt/hailjwt -iname \*.py)
	-mkdir hailjwt
	cp ../hailjwt/setup.py hailjwt/
	rsync -a --prune-empty-dirs --include '*/' --include '*.py' --exclude '*' ../hailjwt/hailjwt/ hailjwt/

install-hailjwt: hailjwt
	cd hailjwt && python3 setup.py install --user

install: install-hailjwt
	pip install -U ./

flake8-stmp: $(PY_FILES) install-hailjwt
	python3 -m flake8 batch
	touch $@

pylint-stmp: $(PY_FILES) install-hailjwt
	python3 -m pylint --rcfile ../pylintrc batch --score=n
	touch $@

check: $(PY_CHECKERS)

build:
	make -C ../docker build
	-docker pull $(BATCH_LATEST)
	mkdir -p hailjwt
	cp ../hailjwt/setup.py hailjwt/
	cp -R ../hailjwt/hailjwt/ hailjwt/hailjwt/
	docker build -t batch --cache-from batch,$(BATCH_LATEST),base,ubuntu:18.04 .

build-test: hailjwt
	docker build -t batch-test -f Dockerfile.test .

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

run-local: run-cloud-sql-proxy-batch-test test/jwt-test-user-token
	HAIL_JWT_SECRET_KEY_FILE=test/jwt-test-secret-key \
		CLOUD_SQL_CONFIG_PATH=batch-secrets/batch-test-cloud-sql-config.json \
		BATCH_USE_KUBE_CONFIG=1 \
		python3 -c 'import batch.server; batch.server.serve('$(BATCH_PORT)')'

test: push-test
	sed -e "s,@image@,$(BATCH_TEST_IMAGE)," \
	  < test-batch-pod.yaml.in > test-batch-pod.yaml
	kubectl create -f test-batch-pod.yaml

ifneq ($(PY_CHECK),false)
test-local: check
endif
test-local: install-cloud-sql-proxy install-hailjwt
test-local: batch-secrets/batch-test-cloud-sql-config.json test/jwt-test-user-token
	rm -rf logs
	HAIL_JWT_SECRET_KEY_FILE=test/jwt-test-secret-key \
		HAIL_TOKEN_FILE=test/jwt-test-user-token \
		POD_NAMESPACE='test' \
		BATCH_USE_KUBE_CONFIG=1 \
		./test-locally.sh

# local means server and test client are two processes on one machine
# in-cluster means in a k8s pod (from which we get k8s creds)
test-local-in-cluster: /batch-secrets/batch-test-cloud-sql-config.json
test-local-in-cluster: check install-hailjwt test/jwt-test-user-token
	HAIL_JWT_SECRET_KEY_FILE=test/jwt-test-secret-key \
		HAIL_TOKEN_FILE=test/jwt-test-user-token \
		POD_NAMESPACE=test \
		./test-locally.sh

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
	rm -rf hailjwt
	rm -f $(PY_CHECKERS)
	rm -rf batch-secrets

clean-db-production-local: batch-secrets/batch-production-cloud-sql-config.json
	CLOUD_SQL_CONFIG_PATH=$< ./clean-db.sh

clean-db-production: /batch-secrets/batch-production-cloud-sql-config.json
	CLOUD_SQL_CONFIG_PATH=$< ./clean-db.sh

clean-db-test-local: batch-secrets/batch-test-cloud-sql-config.json
	CLOUD_SQL_CONFIG_PATH=$< ./clean-db.sh

clean-db-test: /batch-secrets/batch-test-cloud-sql-config.json
	CLOUD_SQL_CONFIG_PATH=$< ./clean-db.sh
