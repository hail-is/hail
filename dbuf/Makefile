.PHONY: push
push:
	docker build -t gcr.io/hail-vdc/dbuf .
	docker push gcr.io/hail-vdc/dbuf
	kubectl -n dk-test delete -f k8s.yaml
	kubectl -n dk-test apply -f k8s.yaml
