.PHONY: build push run-docker run deploy clean-jobs build-worker push-worker

build:
	docker build . -t cronus

push: IMAGE = gcr.io/broad-ctsa/cronus:$(shell docker images -q --no-trunc cronus | sed -e 's,[^:]*:,,')
push: build
	echo $(IMAGE) > cronus-image
	docker tag cronus $(IMAGE)
	docker push $(IMAGE)

build-worker:
	cd worker && docker build . -t cronus-worker

push-worker: IMAGE = gcr.io/broad-ctsa/cronus-worker:$(shell docker images -q --no-trunc cronus-worker | sed -e 's,[^:]*:,,')
push-worker: build-worker
	echo $(IMAGE) > cronus-worker-image
	docker tag cronus-worker $(IMAGE)
	docker push $(IMAGE)

run-docker: build
	docker run -i -p 5000:5000 -t cronus

run:
	python cronus/cronus.py

deploy:
	sed -e "s,@sha@,$(shell git rev-parse --short=12 HEAD)," \
	  -e "s,@image@,$(shell cat cronus-image)," \
	  < deployment.yaml.in > deployment.yaml
	kubectl apply -f deployment.yaml

clean-jobs:
	kubectl delete pods -l app=cronus-job
	kubectl delete services -l app=cronus-job
