.PHONY: build push run-docker run deploy clean clean-workers push-latest-tag

PROJECT = $(shell gcloud config get-value project)
IMAGE_SHORT_NAMES = $(shell ls -1 images)
BUILT = $(foreach image,$(IMAGE_SHORT_NAMES),build/images/$(image)/_BUILT)
PUSHED = $(foreach image,$(IMAGE_SHORT_NAMES),build/images/$(image)/_PUSHED)
PUSHED_LATEST_TAG = $(foreach image,$(IMAGE_SHORT_NAMES),build/images/$(image)/_PUSHED_LATEST_TAG)

.PHONY: $(IMAGE_SHORT_NAMES)

.PHONY: force

build/images/%/_BUILT: force
	mkdir -p $(dir $@)
ifeq ($(CI_BUILD),true)
	-docker pull gcr.io/${PROJECT}/$*:latest
	cd images/$* && docker build . -t $* --cache-from=gcr.io/${PROJECT}/$*:latest
else
	cd images/$* && docker build . -t $*
endif
	touch $@

build/images/%/_PUSHED: IMAGE = gcr.io/$(PROJECT)/$*:$(shell docker images -q --no-trunc $* | sed -e 's,[^:]*:,,')
build/images/%/_PUSHED: build/images/%/_BUILT
	docker tag $* $(IMAGE)
	docker push $(IMAGE)
	echo $(IMAGE) > $@

build/images/%/_PUSHED_LATEST_TAG: IMAGE = gcr.io/$(PROJECT)/$*:latest
build/images/%/_PUSHED_LATEST_TAG: build/images/%/_BUILT
	docker tag $* $(IMAGE)
	docker push $(IMAGE)
	echo $(IMAGE) > $@

$(IMAGE_SHORT_NAMES): %: build/images/%/_BUILT

notebook-worker-images: $(PUSHED)
	cat $^ > $@

build: $(BUILT) notebook-worker-images
ifeq ($(CI_BUILD),true)
	-docker pull gcr.io/${PROJECT}/notebook:latest
	docker build . -t notebook --cache-from=gcr.io/${PROJECT}/notebook:latest
else
	docker build . -t notebook
endif

push: IMAGE = gcr.io/$(PROJECT)/notebook:$(shell docker images -q --no-trunc notebook | sed -e 's,[^:]*:,,')
push: build $(PUSHED)
	docker tag notebook $(IMAGE)
	docker push $(IMAGE)
	echo $(IMAGE) > notebook-image

push-latest-tag: IMAGE = gcr.io/$(PROJECT)/notebook:latest
push-latest-tag: build
	docker tag notebook $(IMAGE)
	docker push $(IMAGE)

push-all-latest-tag: push-latest-tag $(PUSHED_LATEST_TAG)

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
