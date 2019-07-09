PROJECT := $(shell gcloud config get-value project)

UKBB_RG_STATIC_LATEST = gcr.io/$(PROJECT)/ukbb-rg-static:latest
UKBB_RG_STATIC_IMAGE = gcr.io/$(PROJECT)/ukbb-rg-static:$(shell docker images -q --no-trunc ukbb-rg-static | sed -e 's,[^:]*:,,')

UKBB_RG_BROWSER_LATEST = gcr.io/$(PROJECT)/ukbb-rg-browser:latest
UKBB_RG_BROWSER_IMAGE = gcr.io/$(PROJECT)/ukbb-rg-browser:$(shell docker images -q --no-trunc ukbb-rg-browser | sed -e 's,[^:]*:,,')

.PHONY: build
build:
	docker pull ubuntu:18.04
	-docker pull $(UKBB_RG_STATIC_LATEST)
	docker build -t ukbb-rg-static -f Dockerfile.static --cache-from ukbb-rg-static,$(UKBB_RG_STATIC_LATEST),ubuntu:18.04 .
	-docker pull $(UKBB_RG_BROWSER_LATEST)
	docker build -t ukbb-rg-browser -f Dockerfile.browser --cache-from ukbb-rg-browser,$(UKBB_RG_BROWSER_LATEST),ubuntu:18.04 .

.PHONY: push
push: build
	docker tag ukbb-rg-static $(UKBB_RG_STATIC_LATEST)
	docker push $(UKBB_RG_STATIC_LATEST)
	docker tag ukbb-rg-static $(UKBB_RG_STATIC_IMAGE)
	docker push $(UKBB_RG_STATIC_IMAGE)
	docker tag ukbb-rg-browser $(UKBB_RG_BROWSER_LATEST)
	docker push $(UKBB_RG_BROWSER_LATEST)
	docker tag ukbb-rg-browser $(UKBB_RG_BROWSER_IMAGE)
	docker push $(UKBB_RG_BROWSER_IMAGE)

.PHONY: deploy
deploy: push
	kubectl apply -f deployment.yaml
