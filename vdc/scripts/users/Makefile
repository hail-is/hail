.PHONY: check test-local test-local-in-cluster user
		install-cloud-sql-proxy local-cloud-sql-config run-cloud-sql-proxy

include ../../../cloud-sql.mk

PY_CHECKERS = flake8-stmp pylint-stmp

flake8-stmp:
	python3 -m flake8 ./

pylint-stmp:
	python3 -m pylint --rcfile ../../../batch/batch/pylintrc ./ --score=n

check: $(PY_CHECKERS)

test-local: check install-cloud-sql-proxy local-cloud-sql-config run-cloud-sql-proxy
	BATCH_USE_KUBE_CONFIG=1 SQL_HOST=127.0.0.1 pytest

# local means server and test client are two processes on one machine
# in-cluster means in a k8s pod (from which we get k8s creds)
test-local-in-cluster: check
	pytest

user: install-cloud-sql-proxy local-cloud-sql-config run-cloud-sql-proxy
	BATCH_USE_KUBE_CONFIG=1 SQL_HOST=127.0.0.1 python ./user_data.py $(name) $(op)

clean:
	rm -f $(PY_CHECKERS) && rm -rf batch-secrets