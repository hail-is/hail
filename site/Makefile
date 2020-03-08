.PHONY: deployment.yaml build push deploy clean test

PROJECT = $(shell gcloud config get-value project)

IMAGE = gcr.io/$(PROJECT)/site:$(shell docker images -q --no-trunc site | sed -e 's,[^:]*:,,')

build:
	docker build -t site .

test: build
	docker run -p 80:80 -v $${PWD}/../hail/build/www:/var/www/html site nginx -g "daemon off;"

push: build
	docker tag site $(IMAGE)
	docker push $(IMAGE)

deploy: push
	kubectl -n default apply -f service.yaml
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":true,"site_image":{"image":"$(IMAGE)"}}' deployment.yaml deployment.yaml.out
	kubectl -n default apply -f deployment.yaml.out

clean:
	rm -rf deployment.yaml
