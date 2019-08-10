BATCH_GSA_SECRET_NAME ?= gsa-key-n6hf9
BATCH_JWT_SECRET_NAME ?= user-jwt-fh7kp

BATCH_TEST_GSA_SECRET_NAME ?= gsa-key-2x975
BATCH_TEST_JWT_SECRET_NAME ?= user-jwt-vkqfw

batch-secrets/batch-test-gsa-key/privateKeyData:
	mkdir -p batch-secrets/batch-test-gsa-key && kubectl get secret \
	    $(BATCH_TEST_GSA_SECRET_NAME) -n batch-pods \
	    -o "jsonpath={.data.privateKeyData}" \
	  | base64 --decode > $@

batch-secrets/batch-test-jwt/jwt:
	mkdir -p batch-secrets/batch-test-jwt && kubectl get secret \
	    $(BATCH_TEST_JWT_SECRET_NAME) -n batch-pods \
	    -o "jsonpath={.data.jwt}" \
	  | base64 --decode > $@
