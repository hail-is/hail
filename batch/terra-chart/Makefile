include ../../config.mk

.PHONY: batch-image batch-worker-image shadow-jar publish-terra-helm-chart apply-terra-helm-chart

TERRA_BATCH_IMAGE := terradevacrpublic.azurecr.io/hail/batch:$(TOKEN)
TERRA_BATCH_WORKER_IMAGE := terradevacrpublic.azurecr.io/hail/batch-worker:$(TOKEN)
TERRA_IMAGES = pushed-terra-batch-image pushed-terra-batch-worker-image

batch-image:
	$(MAKE) -C ../.. batch-image

batch-worker-image:
	$(MAKE) -C ../.. batch-worker-image CLOUD=azure

shadow-jar:
	$(MAKE) -C ../../hail shadowJar

terra-batch-image: batch-image
	docker build \
		-t $(TERRA_BATCH_IMAGE) \
		-f Dockerfile.batch \
		--build-arg BASE_IMAGE=$(shell cat ../../batch-image) ../..
	echo $(TERRA_BATCH_IMAGE) > $@

terra-batch-worker-image: batch-worker-image shadow-jar
	docker build \
		-t $(TERRA_BATCH_WORKER_IMAGE) \
		-f Dockerfile.worker \
		--build-arg BASE_IMAGE=$(shell cat ../../batch-worker-image)
	echo $(TERRA_BATCH_WORKER_IMAGE) > $@

terra-acr-login:
	# ACR login is short-lived. Log in again if it's been more than an hour
	test -n "$$(find $@-stmp -cmin -60)" || \
	{ az acr login --name terradevacrpublic; touch $@-stmp; }

$(TERRA_IMAGES): pushed-%: % terra-acr-login
	docker push $(shell cat $*)
	echo $(shell cat $*) > $@

values.yaml: $(TERRA_IMAGES)
	python3 ../../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"batch_image":{"image":"$(shell cat pushed-terra-batch-image)"},"batch_worker_image":{"image":"$(shell cat pushed-terra-batch-worker-image)"}}' values.yaml.tmpl values.yaml

publish-terra-helm-chart: values.yaml
	helm package .
	helm push hail-batch-terra-azure-*.tgz oci://terradevacrpublic.azurecr.io/hail

apply-terra-helm-chart: values.yaml
	helm -n $(NAMESPACE) upgrade $(DEPLOYMENT_NAME) .
