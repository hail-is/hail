PROJECT := $(shell gcloud config get-value project)

BATCH_LATEST = gcr.io/$(PROJECT)/batch:latest
BATCH_IMAGE = gcr.io/$(PROJECT)/batch:$(shell docker images -q --no-trunc batch | sed -e 's,[^:]*:,,')

BATCH_TEST_LATEST = gcr.io/$(PROJECT)/batch-test:latest
BATCH_TEST_IMAGE = gcr.io/$(PROJECT)/batch-test:$(shell docker images -q --no-trunc batch-test | sed -e 's,[^:]*:,,')

PY_FILES = $(shell find batch -iname \*.py -not -exec git check-ignore -q {} \; -print)

PYTHONPATH := $${PYTHONPATH:+$${PYTHONPATH}:}../gear
PYTHON := PYTHONPATH=$(PYTHONPATH) python3

flake8-stmp: $(PY_FILES)
	$(PYTHON) -m flake8  --config ../setup.cfg batch
	touch $@

pylint-stmp: $(PY_FILES)
	$(PYTHON) -m pylint --rcfile ../pylintrc batch --score=n
	touch $@

check: flake8-stmp pylint-stmp

build:
	make -C ../docker build
	-docker pull $(BATCH_LATEST)
	python3 ../ci/jinja2_render.py '{"base_image":{"image":"base"}}' Dockerfile Dockerfile.out
	docker build -t batch -f Dockerfile.out --cache-from batch,$(BATCH_LATEST),base ..

push: build
	docker tag batch $(BATCH_LATEST)
	docker push $(BATCH_LATEST)
	docker tag batch $(BATCH_IMAGE)
	docker push $(BATCH_IMAGE)

deploy: push
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":true,"batch_image":{"image":"$(BATCH_IMAGE)"},"batch_pods_ns":{"name":"batch-pods"},"batch_database":{"user_secret_name":"sql-batch-batch-admin-config"},"global":{"domain":"$(DOMAIN)"}}' deployment.yaml deployment.yaml.out
	kubectl -n default apply -f deployment.yaml.out

clean:
	rm -f flake8-stmp pylint-stmp
