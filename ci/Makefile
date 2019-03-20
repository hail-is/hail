.PHONY: run setup-conda-env build-hail-ci push-hail-ci test-locally
.PHONY: run-service deploy test-in-cluster test

PROJECT = $(shell gcloud config get-value project)

setup-conda-env:
	conda env create --name hail-ci -f environment.yml

update-conda-env:
	conda env update --name hail-ci -f environment.yml

build-hail-ci:
	cd ../ && docker build . -t hail-ci -f ci/Dockerfile

push-hail-ci: IMAGE = gcr.io/$(PROJECT)/hail-ci:$(shell docker images -q --no-trunc hail-ci | head -n 1 | sed -e 's,[^:]*:,,')
push-hail-ci: build-hail-ci
	docker tag hail-ci ${IMAGE}
	docker push ${IMAGE}
	echo ${IMAGE} > hail-ci-image
	echo built ${IMAGE}

ci-test:
	./test.sh

test: UUID := $(shell ./generate-uid.sh)
test:
	docker build . -f Dockerfile.test -t hail-ci-test
	docker tag \
				 hail-ci-test \
				 gcr.io/$(PROJECT)/hail-ci-test:$$(docker images -q --no-trunc hail-ci-test | head -n 1 | sed -e 's,[^:]*:,,')
	docker push gcr.io/$(PROJECT)/hail-ci-test:$$(docker images -q --no-trunc hail-ci-test | head -n 1 | sed -e 's,[^:]*:,,')
	sed -e "s,@sha@,$(shell git rev-parse --short=12 HEAD)," \
		-e "s,@image@,gcr.io/$(PROJECT)/hail-ci-test:$$(docker images -q --no-trunc hail-ci-test | head -n 1 | sed -e 's,[^:]*:,,')," \
		-e "s,@uuid@,$(UUID)," \
		< test-pod.yaml.in > test-pod.yaml
	kubectl apply -f test-pod.yaml --namespace batch-pods
	../until-with-fuel 120 kubectl logs hail-ci-test-$(UUID) --namespace batch-pods
	kubectl logs -f hail-ci-test-$(UUID) --namespace batch-pods
	kubectl delete pod hail-ci-test-$(UUID) --namespace batch-pods

run-service:
	kubectl apply -f k8s/service.yaml

deploy: push-hail-ci
	sed -e "s,@sha@,$(shell git rev-parse --short=12 HEAD)," \
	  -e "s,@image@,$(shell cat hail-ci-image)," \
	  < deployment.yaml.in > k8s/deployment.yaml
	kubectl -n default apply -f k8s
