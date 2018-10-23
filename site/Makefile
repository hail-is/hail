.PHONY: deployment.yaml build push deploy clean

PROJECT = $(shell gcloud config get-value project)

IMAGE = gcr.io/$(PROJECT)/site:$(shell docker images -q --no-trunc site | sed -e 's,[^:]*:,,')

deployment.yaml: deployment.yaml.in build
	sed -e "s,@sha@,$$(git rev-parse --short=12 HEAD),g" \
	  -e "s,@image@,$(IMAGE)," \
	  < $< > $@

build:
	docker build -t site .

push: build
	docker tag site $(IMAGE)
	docker push $(IMAGE)

deploy: push deployment.yaml
	kubectl apply -f service.yaml
	kubectl apply -f deployment.yaml

clean:
	rm -rf deployment.yaml
