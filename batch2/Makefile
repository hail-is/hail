PROJECT := $(shell gcloud config get-value project)

BATCH2_LATEST = gcr.io/$(PROJECT)/batch2:latest
BATCH2_IMAGE = gcr.io/$(PROJECT)/batch2:$(shell docker images -q --no-trunc batch2 | sed -e 's,[^:]*:,,')

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
	-docker pull $(BATCH2_LATEST)
	python3 ../ci/jinja2_render.py '{"service_base_image":{"image":"service-base"}}' Dockerfile Dockerfile.out
	docker build -t batch2 -f Dockerfile.out --cache-from batch2,$(BATCH2_LATEST),service-base ..

push: build
	docker tag batch2 $(BATCH2_LATEST)
	docker push $(BATCH2_LATEST)
	docker tag batch $(BATCH2_IMAGE)
	docker push $(BATCH2_IMAGE)

deploy: push
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":true,"batch2_image":{"image":"$(BATCH2_IMAGE)"},"batch_pods_ns":{"name":"batch-pods"},"batch_database":{"user_secret_name":"sql-batch-batch-admin-config"},"global":{"domain":"$(DOMAIN)"}}' deployment.yaml deployment.yaml.out
	kubectl -n default apply -f deployment.yaml.out

clean:
	rm -f flake8-stmp pylint-stmp
