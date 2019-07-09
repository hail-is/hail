PROJECT = $(shell gcloud config get-value project)
AUTH_LATEST = gcr.io/$(PROJECT)/auth:latest
AUTH_IMAGE = gcr.io/$(PROJECT)/auth:$(shell docker images -q --no-trunc auth:latest | sed -e 's,[^:]*:,,')

.PHONY: build
build:
	make -C ../docker build
	-docker pull $(AUTH_LATEST)
	python3 ../ci/jinja2_render.py '{"base_image":{"image":"base"}}' Dockerfile Dockerfile.out
	docker build -f Dockerfile.out -t auth --cache-from auth,$(AUTH_LATEST),base ..

.PHONY: push
push: build
	docker tag auth $(AUTH_LATEST)
	docker push $(AUTH_LATEST)
	docker tag auth $(AUTH_IMAGE)
	docker push $(AUTH_IMAGE)

.PHONY: deploy
deploy: push
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":true,"auth_image":{"image":"$(AUTH_IMAGE)"}}' deployment.yaml deployment.yaml.out
	kubectl -n default apply -f deployment.yaml.out
