PROJECT := $(shell gcloud config get-value project)

UKBB_RG_STATIC_LATEST = gcr.io/$(PROJECT)/ukbb-rg-static:latest
UKBB_RG_STATIC_IMAGE = gcr.io/$(PROJECT)/ukbb-rg-static:$(shell docker images -q --no-trunc ukbb-rg-static | sed -e 's,[^:]*:,,')

UKBB_RG_BROWSER_LATEST = gcr.io/$(PROJECT)/ukbb-rg-browser:latest
UKBB_RG_BROWSER_IMAGE = gcr.io/$(PROJECT)/ukbb-rg-browser:$(shell docker images -q --no-trunc ukbb-rg-browser | sed -e 's,[^:]*:,,')

.PHONY: build
build:
	ls app/app.R app/www/rainbowvis.js # read the README
	docker pull ubuntu:18.04
	-docker pull $(UKBB_RG_STATIC_LATEST)
	docker build -t ukbb-rg-static -f Dockerfile.static --cache-from ukbb-rg-static,$(UKBB_RG_STATIC_LATEST),ubuntu:18.04 .
	docker pull ubuntu:19.04
	-docker pull $(UKBB_RG_BROWSER_LATEST)
	docker build -t ukbb-rg-browser -f Dockerfile.browser --cache-from ukbb-rg-browser,$(UKBB_RG_BROWSER_LATEST),ubuntu:19.04 .

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

update-ukbb-rg-browser-files:
	mkdir -p data
	mkdir -p data/Rdata_outputs
	[ -e data/Rdata_outputs ] \
    || gsutil -m cp -R gs://rg-website/Rdata_outputs/ data/Rdata_outputs
	kubectl exec ukbb-rg-browser-0 -n ukbb-rg -- /bin/sh -c '[ -e /ukbb-rg-browser/Rdata_outputs ]' \
    || kubectl cp Rdata_outputs ukbb-rg/ukbb-rg-browser:/ukbb-rg-browser/Rdata_outputs

update-ukbb-rg-static-files:
	mkdir -p data
	mkdir -p data/docs
	[ -e data/docs ] \
    || gsutil -m cp -R gs://rg-website/docs/ data/docs
	mkdir -p data/site_libs
	[ -e data/site_libs ] \
    || gsutil -m cp -R gs://rg-website/site_libs/ data/site_libs # >2GB of data
	kubectl exec ukbb-rg-static-0 -n ukbb-rg -- /bin/sh -c '[ -e /ukbb-rg-static/rg_summary_102.html ]' \
    || kubectl cp docs/rg_summary_102.html ukbb-rg/ukbb-rg-static:/ukbb-rg-static/rg_summary_102.html
	kubectl exec ukbb-rg-static-0 -n ukbb-rg -- /bin/sh -c '[ -e /ukbb-rg-static/site_libs ]' \
    || kubectl cp site_libs ukbb-rg/ukbb-rg-static:/ukbb-rg-static/site_libs
