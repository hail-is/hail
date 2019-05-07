.PHONY: test-local test-local-in-cluster check clean

PYTHONPATH := $${PYTHONPATH:+$${PYTHONPATH}:}../batch:../hailjwt
PYTHON := PYTHONPATH=$(PYTHONPATH) python3

PY_FILES = $(shell find pipeline -iname \*.py -not -exec git check-ignore -q {} \; -print)

include ../cloud-sql.mk
include ../batch/batch-secrets.mk

flake8-stmp: $(PY_FILES)
	$(PYTHON) -m flake8 pipeline
	touch $@

pylint-stmp: $(PY_FILES)
	$(PYTHON) -m pylint --rcfile pipeline/pylintrc pipeline --score=n
	touch $@

check: flake8-stmp pylint-stmp

../batch/test/jwt-test-user-token:
	make -C ../batch test/jwt-test-user-token

test-local: install-cloud-sql-proxy
test-local: batch-secrets/batch-test-cloud-sql-config.json
test-local: batch-secrets/batch-test-gsa-key/privateKeyData
test-local: batch-secrets/batch-test-jwt/jwt
test-local: ../batch/test/jwt-test-user-token check
	HAIL_JWT_SECRET_KEY_FILE=../batch/test/jwt-test-secret-key \
		HAIL_TOKEN_FILE=../batch/test/jwt-test-user-token \
		HAIL_POD_NAMESPACE=test \
		BATCH_USE_KUBE_CONFIG=1 \
		CLOUD_SQL_PROXY=1 \
		PYTHONPATH=$(PYTHONPATH) \
		./test-locally.sh

# local means server and test client are two processes on one machine
# in-cluster means in a k8s pod (from which we get k8s creds)
test-local-in-cluster: ../batch/test/jwt-test-user-token check
	HAIL_JWT_SECRET_KEY_FILE=../batch/test/jwt-test-secret-key \
		HAIL_TOKEN_FILE=../batch/test/jwt-test-user-token \
		HAIL_POD_NAMESPACE=test \
		PYTHONPATH=$(PYTHONPATH) \
		./test-locally.sh

clean:
	rm -f flake8-stmp pylint-stmp
