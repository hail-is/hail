PROJECT = $(shell gcloud config get-value project)
DOMAIN ?= hail.is

BENCHMARK_LATEST = gcr.io/$(PROJECT)/benchmark:latest
BENCHMARK_IMAGE = gcr.io/$(PROJECT)/benchmark:$(shell docker images -q --no-trunc benchmark:latest | sed -e 's,[^:]*:,,')

PYTHONPATH := $${PYTHONPATH:+$${PYTHONPATH}:}
PYTHON := PYTHONPATH=$(PYTHONPATH)../hail/python:../gear:../web_common python3

.PHONY: check
check:
	$(PYTHON) -m flake8 benchmark
	$(PYTHON) -m pylint --rcfile ../pylintrc benchmark --score=n

.PHONY: build
build:
	make -C ../docker build
	-docker pull $(BENCHMARK_LATEST)
	python3 ../ci/jinja2_render.py '{"service_base_image":{"image":"service-base"}}' Dockerfile Dockerfile.out
	docker build -f Dockerfile.out -t benchmark --cache-from benchmark,$(BENCHMARK_LATEST),service-base ..

.PHONY: push
push: build
	docker tag benchmark $(BENCHMARK_LATEST)
	docker push $(BENCHMARK_LATEST)
	docker tag benchmark $(BENCHMARK_IMAGE)
	docker push $(BENCHMARK_IMAGE)

.PHONY: deploy
deploy: push
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":true,"default_ns":{"name":"default"},"batch_pods_ns":{"name":"batch-pods"}, "benchmark_image":{"image":"$(BENCHMARK_IMAGE)"},"benchmark_database":{"user_secret_name":"sql-benchmark-user-config"},"global":{"project":"$(PROJECT)","zone":"$(ZONE)","domain":"$(DOMAIN)"}}' deployment.yaml deployment.yaml.out
	kubectl -n default apply -f deployment.yaml.out

.PHONY: clean
clean:
	rm -f Dockerfile.out deployment.yaml.out
