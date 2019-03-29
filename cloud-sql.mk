.PHONY: install-cloud-sql-proxy

CLOUD_SQL_PORT ?= 3306

install-cloud-sql-proxy:
	test -f cloud_sql_proxy || \
		curl -o cloud_sql_proxy https://dl.google.com/cloudsql/cloud_sql_proxy.darwin.amd64 && \
		chmod +x cloud_sql_proxy

local-cloud-sql-config:
	mkdir -p batch-secrets && kubectl get secret \
	    batch-test-cloud-sql-config -n batch-pods \
	    -o "jsonpath={.data.batch-test-cloud-sql-config\.json}" \
	  | base64 --decode \
	  | jq '.host = "127.0.0.1" | .port = $(CLOUD_SQL_PORT)' \
	    > batch-secrets/batch-test-cloud-sql-config.json

run-cloud-sql-proxy: local-cloud-sql-config install-cloud-sql-proxy
	$(eval config := $(shell pwd)/batch-secrets/batch-test-cloud-sql-config.json)
	$(eval connection_name := $(shell jq -r '.connection_name' $(config)))
	$(eval port := $(shell jq -r '.port' $(config)))
	./cloud_sql_proxy -instances=$(connection_name)=tcp:$(port) &
