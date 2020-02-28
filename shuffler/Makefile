PROJECT = $(shell gcloud config get-value project)
DOMAIN ?= hail.is

SHUFFLER_LATEST = gcr.io/$(PROJECT)/shuffler:latest
SHUFFLER_IMAGE = gcr.io/$(PROJECT)/shuffler:$(shell docker images -q --no-trunc shuffler:latest | sed -e 's,[^:]*:,,')

TEST_SHUFFLER_IMAGE = gcr.io/$(PROJECT)/test_shuffler:$(shell docker images -q --no-trunc test_shuffler:latest | sed -e 's,[^:]*:,,')

PYTHONPATH := $${PYTHONPATH:+$${PYTHONPATH}:}
PYTHON := PYTHONPATH=$(PYTHONPATH)../hail/python:../gear:../web_common python3

.PHONY: check
check:
	$(PYTHON) -m flake8 shuffler
	$(PYTHON) -m pylint --rcfile ../pylintrc shuffler --score=n

.PHONY: build
build:
	make -C ../docker build
	-docker pull $(SHUFFLER_LATEST)
	python3 ../ci/jinja2_render.py '{"service_base_image":{"image":"service-base"}}' Dockerfile Dockerfile.out
	mkdir context
	cp target/scala-2.11/shuffler-assembly-0.1.jar context
	docker build -f Dockerfile.out -t shuffler --cache-from shuffler,$(SHUFFLER_LATEST),base context
	rm -rf context

.PHONY: push
push: build
	docker tag shuffler $(SHUFFLER_LATEST)
	docker push $(SHUFFLER_LATEST)
	docker tag shuffler $(SHUFFLER_IMAGE)
	docker push $(SHUFFLER_IMAGE)

.PHONY: deploy
deploy: push
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":true,"shuffler_image":{"image":"$(SHUFFLER_IMAGE)"},"global":{"project":"$(PROJECT)","zone":"$(ZONE)","domain":"$(DOMAIN)"}}' deployment.yaml deployment.yaml.out
	kubectl -n dking apply -f deployment.yaml.out

.PHONY:
clean:
	rm -f Dockerfile.out deployment.yaml.out
