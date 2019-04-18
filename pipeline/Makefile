.PHONY: test-local test-local-in-cluster check clean

PYTHON := PYTHONPATH=$${PYTHONPATH:+$${PYTHONPATH}:}../batch python3

PY_FILES = $(shell find pipeline -iname \*.py -not -exec git check-ignore -q {} \; -print)

include ../cloud-sql.mk

flake8-stmp: $(PY_FILES)
	$(PYTHON) -m flake8 pipeline
	touch $@

pylint-stmp: $(PY_FILES)
	$(PYTHON) -m pylint --rcfile pipeline/pylintrc pipeline --score=n
	touch $@

check: flake8-stmp pylint-stmp

test-local: install-cloud-sql-proxy batch-secrets/batch-test-cloud-sql-config.json check
	POD_NAMESPACE='test' BATCH_USE_KUBE_CONFIG=1 ./test-locally.sh

# local means server and test client are two processes on one machine
# in-cluster means in a k8s pod (from which we get k8s creds)
test-local-in-cluster: check
	POD_NAMESPACE='test' ./test-locally.sh

clean:
	rm -f flake8-stmp pylint-stmp
