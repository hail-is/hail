.PHONY: deployment.yaml build push deploy clean test

PROJECT = $(shell gcloud config get-value project)

IMAGE = gcr.io/$(PROJECT)/site:$(shell docker images -q --no-trunc site | sed -e 's,[^:]*:,,')

build:
	make -C www
	-rm -rf www-out
	rsync --archive \
        --verbose \
        --exclude='*.xslt' \
        --exclude='*.md' \
        www/* \
        www-out
	-rm ../site-www.tar.gz
	cd www-out && tar -cvzf ../site-www.tar.gz *
	docker build -t site .

push: build
	docker tag site $(IMAGE)
	docker push $(IMAGE)

deploy: push
	test $(NAMESPACE)
	python3 ../ci/jinja2_render.py '{"global": {"domain":"hail.is"},"default_ns":{"name":"$(NAMESPACE)"},"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":true,"site_image":{"image":"$(IMAGE)"}}' deployment.yaml deployment.yaml.out
	kubectl -n $(NAMESPACE) apply -f deployment.yaml.out

clean:
	rm -rf deployment.yaml
