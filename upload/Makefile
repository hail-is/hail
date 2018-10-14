.PHONY: build push redeploy

PROJECT = $(shell gcloud config get-value project)

build:
	docker build . -t upload

push: IMAGE="gcr.io/$(PROJECT)/upload:$(shell docker images -q --no-trunc upload | sed -e 's,[^:]*:,,')"
push: build
	docker tag upload $(IMAGE)
	docker push $(IMAGE)
	echo $(IMAGE) > upload-image

deploy:
	sed -e "s,@sha@,$$(git rev-parse --short=12 HEAD)," \
	  -e "s,@image@,$$(cat upload-image)," \
	  < deployment.yaml.in > deployment.yaml
	kubectl apply -f deployment.yaml
