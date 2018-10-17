.PHONY: build push deploy

images: ../notebook/notebook-worker-image
	cat $< > images

build: images
	docker build . -t image-fetcher

push: IMAGE = gcr.io/broad-ctsa/image-fetcher:$(shell docker images -q --no-trunc image-fetcher | sed -e 's,[^:]*:,,')
push: build
	echo $(IMAGE) > image-fetcher-image
	docker tag image-fetcher $(IMAGE)
	docker push $(IMAGE)

deploy: push
	sed -e "s,@sha@,$(shell git rev-parse --short=12 HEAD)," \
	  -e "s,@image@,$(shell cat image-fetcher-image)," \
	  < deployment.yaml.in > deployment.yaml
	kubectl apply -f deployment.yaml
