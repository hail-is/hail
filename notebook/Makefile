.PHONY: build push run-docker run deploy clean clean-workers

PROJECT = $(shell gcloud config get-value project)
BUILT = $(shell ls -1 images | awk '{print "build/images/" $$0 "/_BUILT"}')
PUSHED = $(shell ls -1 images | awk '{print "build/images/" $$0 "/_PUSHED"}')

build/images/%/_BUILT: images/%/Dockerfile $(shell find images/$*)
	mkdir -p $(dir $@)
	cd images/$* && docker build . -t $*
	touch $@

build/images/%/_PUSHED: IMAGE = gcr.io/$(PROJECT)/$*:$(shell docker images -q --no-trunc $* | sed -e 's,[^:]*:,,')
build/images/%/_PUSHED: build/images/%/_BUILT
	docker tag $* $(IMAGE)
	docker push $(IMAGE)
	echo $(IMAGE) > $@

notebook-worker-images: $(PUSHED)
	cat $^ > $@

build: $(BUILT) notebook-worker-images
	echo $(BUILT)
	docker build . -t notebook

push: IMAGE = gcr.io/$(PROJECT)/notebook:$(shell docker images -q --no-trunc notebook | sed -e 's,[^:]*:,,')
push: build $(PUSHED)
	echo $(IMAGE) > notebook-image
	docker tag notebook $(IMAGE)
	docker push $(IMAGE)

run-docker: build
	docker run -i -p 5000:5000 -t notebook

run:
	python notebook/notebook.py

deploy: push
	sed -e "s,@sha@,$(shell git rev-parse --short=12 HEAD)," \
	  -e "s,@image@,$(shell cat notebook-image)," \
	  < deployment.yaml.in > deployment.yaml
	kubectl -n default apply -f deployment.yaml

clean-workers:
	kubectl delete pods -l app=notebook-worker
	kubectl delete services -l app=notebook-worker

clean:
	rm -rf build