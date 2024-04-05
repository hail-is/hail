include ../../config.mk
include ../../hail/version.mk

.PHONY: batch-image batch-worker-image shadow-jar publish-terra-helm-chart apply-terra-helm-chart

TERRA_BATCH_IMAGE := terradevacrpublic.azurecr.io/hail/batch:$(TOKEN)
TERRA_BATCH_WORKER_IMAGE := terradevacrpublic.azurecr.io/hail/batch-worker:$(TOKEN)
TERRA_IMAGES = pushed-terra-batch-image pushed-terra-batch-worker-image

terra-batch-image:
	$(MAKE) -C ../.. terra-batch-image IMAGE_NAME=$(TERRA_BATCH_IMAGE)

terra-batch-worker-image:
	$(MAKE) -C ../.. terra-batch-worker-image IMAGE_NAME=$(TERRA_BATCH_WORKER_IMAGE) CLOUD=azure

terra-acr-login:
	# ACR login is short-lived. Log in again if it's been more than an hour
	test -n "$$(find $@-stmp -cmin -60)" || \
	{ az acr login --name terradevacrpublic && touch $@-stmp; }

$(TERRA_IMAGES): pushed-%: terra-acr-login %
	docker push $(shell cat ../../$*)
	echo $(shell cat ../../$*) > $@

values.yaml: $(TERRA_IMAGES)
	python3 ../../ci/jinja2_render.py '{"hail_version":"$(HAIL_PIP_VERSION)"}' Chart.yaml.tmpl Chart.yaml
	python3 ../../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"batch_image":{"image":"$(shell cat pushed-terra-batch-image)"},"batch_worker_image":{"image":"$(shell cat pushed-terra-batch-worker-image)"}}' values.yaml.tmpl values.yaml

publish-terra-helm-chart: values.yaml
	helm package .
	helm push hail-batch-terra-azure-*.tgz oci://terradevacrpublic.azurecr.io/hail

apply-terra-helm-chart: values.yaml
	helm -n $(NAMESPACE) upgrade $(DEPLOYMENT_NAME) .
