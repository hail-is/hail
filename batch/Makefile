.PHONY: build build-test \
  push push-test \
  run-docker run \
  test test-local deploy test-deploy \
  check clean

PROJECT := $(shell gcloud config get-value project)
BATCH_LATEST = gcr.io/$(PROJECT)/batch:latest
BATCH_IMAGE = gcr.io/$(PROJECT)/batch:$(shell docker images -q --no-trunc batch | sed -e 's,[^:]*:,,')

BATCH_TEST_LATEST = gcr.io/$(PROJECT)/batch-test:latest
BATCH_TEST_IMAGE = gcr.io/$(PROJECT)/batch-test:$(shell docker images -q --no-trunc batch-test | sed -e 's,[^:]*:,,')

PY_FILES = $(shell find batch -iname \*.py -not -exec git check-ignore -q {} \; -print)

PYTHONPATH := ../hailjwt$${PYTHONPATH:+:$${PYTHONPATH}}
PYTHON := PYTHONPATH=$(PYTHONPATH) python3

BATCH_PORT ?= 5000

include ../cloud-sql.mk
include batch-secrets.mk

test/jwt-test-user-token: test/jwt-test-user.json test/jwt-test-secret-key create_key.py
	$(PYTHON) create_key.py test/jwt-test-secret-key < $< > $@

install:
	pip install -U ./

flake8-stmp: $(PY_FILES)
	$(PYTHON) -m flake8 batch
	touch $@

pylint-stmp: $(PY_FILES)
	$(PYTHON) -m pylint --rcfile ../pylintrc batch --score=n
	touch $@

check: flake8-stmp pylint-stmp

build:
	make -C ../docker build
	-docker pull $(BATCH_LATEST)
	python3 ../ci/jinja2_render.py '{"base_image":{"image":"base"}}' Dockerfile Dockerfile.out
	docker build -t batch -f Dockerfile.out --cache-from batch,$(BATCH_LATEST),base ..

build-test:
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
	$(PYTHON) -c 'import batch.server; batch.server.serve('$(BATCH_PORT)')'

run-local: run-cloud-sql-proxy-batch-test test/jwt-test-user-token
run-local: batch-secrets/batch-test-gsa-key/privateKeyData
run-local: batch-secrets/batch-test-jwt/jwt
	HAIL_JWT_SECRET_KEY_FILE=test/jwt-test-secret-key \
		CLOUD_SQL_CONFIG_PATH=batch-secrets/batch-test-cloud-sql-config.json \
                GOOGLE_APPLICATION_CREDENTIALS=batch-secrets/batch-test-gsa-key/privateKeyData \
                BATCH_JWT=batch-secrets/batch-test-jwt/jwt \
		BATCH_USE_KUBE_CONFIG=1 \
		$(PYTHON) -c 'import batch.server; batch.server.serve('$(BATCH_PORT)')'

test: push-test
	sed -e "s,@image@,$(BATCH_TEST_IMAGE)," \
	  < test-batch-pod.yaml.in > test-batch-pod.yaml
	kubectl create -f test-batch-pod.yaml

ifneq ($(PY_CHECK),false)
test-local: check
endif
test-local: install-cloud-sql-proxy
test-local: batch-secrets/batch-test-cloud-sql-config.json test/jwt-test-user-token
test-local: batch-secrets/batch-test-gsa-key/privateKeyData
test-local: batch-secrets/batch-test-jwt/jwt
	HAIL_JWT_SECRET_KEY_FILE=test/jwt-test-secret-key \
		HAIL_TOKEN_FILE=test/jwt-test-user-token \
		HAIL_POD_NAMESPACE='test' \
		BATCH_USE_KUBE_CONFIG=1 \
		PYTHONPATH="$(PYTHONPATH)" \
		./test-locally.sh

# local means server and test client are two processes on one machine
# in-cluster means in a k8s pod (from which we get k8s creds)
test-local-in-cluster: check test/jwt-test-user-token
	HAIL_JWT_SECRET_KEY_FILE=test/jwt-test-secret-key \
		HAIL_TOKEN_FILE=test/jwt-test-user-token \
		HAIL_POD_NAMESPACE=test \
		PYTHONPATH="$(PYTHONPATH)" \
		./test-locally.sh

deploy: push
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":true,"batch_image":{"image":"$(BATCH_IMAGE)"},"batch_pods_ns":{"name":"batch-pods"},"batch_database":{"user_secret_name":"sql-batch-batch-admin-config"}}' deployment.yaml deployment.yaml.out
	kubectl -n default apply -f deployment.yaml.out

test-deploy: push-test
	ifndef TEST_NAMESPACE
	$(error TEST_NAMESPACE is not set)
	endif
	sed -e "s,@sha@,$$(git rev-parse --short=12 HEAD)," \
	  -e "s,@image@,$(BATCH_TEST_IMAGE)," \
          -e "s,@gsa_secret@,$(BATCH_TEST_GSA_SECRET_NAME)," \
          -e "s,@jwt_secret@,$(BATCH_TEST_JWT_SECRET_NAME)," \
	  < deployment.yaml.in > deployment.yaml
	kubectl -n $(TEST_NAMESPACE) apply -f deployment.yaml

clean:
	rm -rf hailjwt
	rm -f $(PY_CHECKERS)
	rm -rf batch-secrets

clean-db-production-local: batch-secrets/batch-production-cloud-sql-config.json
	CLOUD_SQL_CONFIG_PATH=$< ./clean-db.sh

clean-db-production:
	CLOUD_SQL_CONFIG_PATH=/batch-secrets/batch-production-cloud-sql-config.json ./clean-db.sh

clean-db-test-local: batch-secrets/batch-test-cloud-sql-config.json
	CLOUD_SQL_CONFIG_PATH=$< ./clean-db.sh

clean-db-test:
	CLOUD_SQL_CONFIG_PATH=/batch-secrets/batch-test-cloud-sql-config.json ./clean-db.sh
