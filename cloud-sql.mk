.PHONY: install-cloud-sql-proxy local-cloud-sql-config run-cloud-sql-proxy

CLOUD_SQL_PORT ?= 3306

install-cloud-sql-proxy:
	test -f cloud_sql_proxy || \
		curl -o cloud_sql_proxy https://dl.google.com/cloudsql/cloud_sql_proxy.darwin.amd64 && \
		chmod +x cloud_sql_proxy

batch-secrets/batch-test-cloud-sql-config.json:
	mkdir -p batch-secrets && kubectl get secret \
	    batch-test-cloud-sql-config -n batch-pods \
	    -o "jsonpath={.data.batch-test-cloud-sql-config\.json}" \
	  | base64 --decode \
	  | jq '.host = "127.0.0.1" | .port = $(CLOUD_SQL_PORT)' \
	    > $@

batch-secrets/batch-production-cloud-sql-config.json:
	mkdir -p batch-secrets && kubectl get secret \
	    batch-production-cloud-sql-config -n default \
	    -o "jsonpath={.data.batch-production-cloud-sql-config\.json}" \
	  | base64 --decode \
	  | jq '.host = "127.0.0.1" | .port = $(CLOUD_SQL_PORT)' \
	    > $@

run-cloud-sql-proxy-batch-test: batch-secrets/batch-test-cloud-sql-config.json
run-cloud-sql-proxy-batch-test: install-cloud-sql-proxy
	$(eval config := $(shell pwd)/batch-secrets/batch-test-cloud-sql-config.json)
	$(eval connection_name := $(shell jq -r '.connection_name' $(config)))
	$(eval port := $(shell jq -r '.port' $(config)))
	./cloud_sql_proxy -instances=$(connection_name)=tcp:$(port) &

run-cloud-sql-proxy-batch-production: batch-secrets/batch-production-cloud-sql-config.json
run-cloud-sql-proxy-batch-production: install-cloud-sql-proxy
	$(eval config := $(shell pwd)/batch-secrets/batch-production-cloud-sql-config.json)
	$(eval connection_name := $(shell jq -r '.connection_name' $(config)))
	$(eval port := $(shell jq -r '.port' $(config)))
	./cloud_sql_proxy -instances=$(connection_name)=tcp:$(port) &
