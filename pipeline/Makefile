.PHONY: test-local test-local-in-cluster

test-local:
	POD_NAMESPACE='test' BATCH_USE_KUBE_CONFIG=1 ./test-locally.sh

# local means server and test client are two processes on one machine
# in-cluster means in a k8s pod (from which we get k8s creds)
test-local-in-cluster:
	POD_NAMESPACE='test' ./test-locally.sh
