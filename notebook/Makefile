.PHONY: build push run-docker run deploy clean-workers build-worker push-worker

build: push-worker
	docker build . -t notebook

push: IMAGE = gcr.io/broad-ctsa/notebook:$(shell docker images -q --no-trunc notebook | sed -e 's,[^:]*:,,')
push: build
	echo $(IMAGE) > notebook-image
	docker tag notebook $(IMAGE)
	docker push $(IMAGE)

build-worker:
	cd worker && docker build . -t notebook-worker

push-worker: IMAGE = gcr.io/broad-ctsa/notebook-worker:$(shell docker images -q --no-trunc notebook-worker | sed -e 's,[^:]*:,,')
push-worker: build-worker
	echo $(IMAGE) > notebook-worker-image
	docker tag notebook-worker $(IMAGE)
	docker push $(IMAGE)

run-docker: build build-worker
	docker run -i -p 5000:5000 -t notebook

run:
	python notebook/notebook.py

deploy: push push-worker
	sed -e "s,@sha@,$(shell git rev-parse --short=12 HEAD)," \
	  -e "s,@image@,$(shell cat notebook-image)," \
	  < deployment.yaml.in > deployment.yaml
	kubectl apply -f deployment.yaml

clean-workers:
	kubectl delete pods -l app=notebook-worker
	kubectl delete services -l app=notebook-worker
