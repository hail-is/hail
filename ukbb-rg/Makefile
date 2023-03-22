include ../config.mk

UKBB_RG_STATIC_IMAGE := $(DOCKER_PREFIX)/ukbb-rg-static:$(TOKEN)
UKBB_RG_BROWSER_IMAGE := $(DOCKER_PREFIX)/ukbb-rg-browser:$(TOKEN)

.PHONY: build
build:
	ls app/app.R app/www/rainbowvis.js # read the README

	python3 ../ci/jinja2_render.py '{"docker_prefix":"$(DOCKER_PREFIX)"}' Dockerfile.static Dockerfile.static.out
	../docker-build.sh . Dockerfile.static.out $(UKBB_RG_STATIC_IMAGE)

	python3 ../ci/jinja2_render.py '{"docker_prefix":"$(DOCKER_PREFIX)"}' Dockerfile.browser Dockerfile.browser.out
	../docker-build.sh . Dockerfile.browser.out $(UKBB_RG_BROWSER_IMAGE)

.PHONY: deploy
deploy: build
	python3 ../ci/jinja2_render.py '{"docker_prefix":"$(DOCKER_PREFIX)","static_image":"$(UKBB_RG_STATIC_IMAGE)","browser_image":"$(UKBB_RG_BROWSER_IMAGE)"}' deployment.yaml deployment.yaml.out
	kubectl apply -n ukbb-rg -f deployment.yaml.out

update-ukbb-rg-browser-files:
	mkdir -p data
	mkdir -p data/Rdata_outputs
	[ -e data/Rdata_outputs ] \
    || gcloud storage cp -r gs://rg-website/Rdata_outputs/ data/Rdata_outputs
	kubectl exec ukbb-rg-browser-0 -n ukbb-rg -- /bin/sh -c '[ -e /ukbb-rg-browser/Rdata_outputs ]' \
    || kubectl cp Rdata_outputs ukbb-rg/ukbb-rg-browser:/ukbb-rg-browser/Rdata_outputs

update-ukbb-rg-static-files:
	mkdir -p data
	mkdir -p data/docs
	[ -e data/docs ] \
    || gcloud storage cp -r gs://rg-website/docs/ data/docs
	mkdir -p data/site_libs
	[ -e data/site_libs ] \
    || gcloud storage cp -r gs://rg-website/site_libs/ data/site_libs # >2GB of data
	kubectl exec ukbb-rg-static-0 -n ukbb-rg -- /bin/sh -c '[ -e /ukbb-rg-static/rg_summary_102.html ]' \
    || kubectl cp docs/rg_summary_102.html ukbb-rg/ukbb-rg-static:/ukbb-rg-static/rg_summary_102.html
	kubectl exec ukbb-rg-static-0 -n ukbb-rg -- /bin/sh -c '[ -e /ukbb-rg-static/site_libs ]' \
    || kubectl cp site_libs ukbb-rg/ukbb-rg-static:/ukbb-rg-static/site_libs
