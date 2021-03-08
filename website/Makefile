include ../config.mk

.PHONY: docs build run run-docker push deploy clean

IMAGE = gcr.io/$(PROJECT)/website:$(shell docker images -q --no-trunc website | sed -e 's,[^:]*:,,')

docs:
	$(MAKE) -C ../hail hail-docs-no-test batch-docs
	tar czf docs.tar.gz -C ../hail/build/www .

build: docs
	$(MAKE) -C ../docker service-base
	python3 ../ci/jinja2_render.py '{"service_base_image": {"image":"service-base"}}' Dockerfile Dockerfile.out
	docker build -f Dockerfile.out -t website .

run: docs
	cd website && tar -xvzf ../docs.tar.gz
	HAIL_DOMAIN=localhost:5000 python3 -m website local

run-docker: build
	docker run -e HAIL_DOMAIN=localhost:5000 -p 5000:5000 website python3 -m website local

push: build
	docker tag website $(IMAGE)
	docker push $(IMAGE)

deploy: push
	! [ -z $(NAMESPACE) ]  # call this like: make deploy NAMESPACE=default
	$(MAKE) -C ../docker hail-ubuntu
	python3 ../ci/jinja2_render.py '{"global": {"domain":"$(DOMAIN)"},"default_ns":{"name":"$(NAMESPACE)"},"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":$(DEPLOY),"website_image":{"image":"$(IMAGE)"}' deployment.yaml deployment.yaml.out
	kubectl -n $(NAMESPACE) apply -f deployment.yaml.out

clean:
	rm -rf deployment.yaml
	rm -rf website/docs
	rm -rf website/static/css
