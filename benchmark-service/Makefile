include ../config.mk

BENCHMARK_LATEST = $(DOCKER_PREFIX)/benchmark:latest
BENCHMARK_IMAGE = $(DOCKER_PREFIX)/benchmark:$(shell docker images -q --no-trunc benchmark:latest | sed -e 's,[^:]*:,,')

EXTRA_PYTHONPATH := ../hail/python:../gear:../web_common
PYTHON := PYTHONPATH=$${PYTHONPATH:+$${PYTHONPATH}:}$(EXTRA_PYTHONPATH) python3

.PHONY: check
check:
	$(PYTHON) -m flake8 benchmark
	$(PYTHON) -m pylint --rcfile ../pylintrc benchmark --score=n

.PHONY: build
build:
	$(MAKE) -C ../docker build
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
	! [ -z $(NAMESPACE) ]  # call this like: make deploy NAMESPACE=default
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":$(DEPLOY),"default_ns":{"name":"$(NAMESPACE)"}, "benchmark_image":{"image":"$(BENCHMARK_IMAGE)"},"benchmark_database":{"user_secret_name":"sql-benchmark-user-config"},"global":{"project":"$(PROJECT)","zone":"$(ZONE)","domain":"$(DOMAIN)"}}' deployment.yaml deployment.yaml.out
	kubectl -n $(NAMESPACE) apply -f deployment.yaml.out

.PHONY: clean
clean:
	rm -f Dockerfile.out deployment.yaml.out
