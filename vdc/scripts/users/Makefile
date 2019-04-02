.PHONY: check test-local test-in-cluster user secret clean flake8-stmp pylint-stmp
		install-cloud-sql-proxy local-cloud-sql-config run-cloud-sql-proxy

include ../../../cloud-sql.mk

INSTANCE = $(shell kubectl get secret create-users -o json | jq -r ".data.instance" | base64 -D)

flake8-stmp:
	python3 -m flake8 ./

pylint-stmp:
	python3 -m pylint --rcfile ../../../batch/batch/pylintrc ./ --score=n

check: flake8-stmp pylint-stmp

instance.cloudsql:
	echo "$(INSTANCE)" > $@

test-local: check install-cloud-sql-proxy instance.cloudsql
	./test-locally.sh

# local means server and test client are two processes on one machine
# in-cluster means in a k8s pod (from which we get k8s creds)
test-in-cluster: check
	python3 -m pytest

user: install-cloud-sql-proxy instance.cloudsql
	./run-locally.sh $(name) $(op)

user-in-cluster:
	python3 ./user_data.py $(name) $(op)

secret:
	./make-secret.sh $(user) $(pass) $(db) $(host) $(instance)

clean:
	rm -rf instance.cloudsql