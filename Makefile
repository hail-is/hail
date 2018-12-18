.PHONY: hail-ci-build-image push-hail-ci-build-image
.DEFAULT_GOAL := default

PROJECT = $(shell gcloud config get-value project)

# We need to pull each image we use as a cache and we must insure the FROM image
# is in our cache-from list. I also include the local `hail-pr-builder` to
# ensure that local iteration on the Dockerfile also uses the cache.
# https://github.com/moby/moby/issues/20316#issuecomment-358260810
hail-ci-build-image:
	-docker pull debian:9.5
	-docker pull $(shell cat hail-ci-build-image)
	docker build . -t hail-pr-builder -f Dockerfile.pr-builder \
	    --cache-from $(shell cat hail-ci-build-image),hail-pr-builder,debian:9.5

push-hail-ci-build-image: hail-ci-build-image
	echo "gcr.io/$(PROJECT)/hail-pr-builder:`docker images -q --no-trunc hail-pr-builder:latest | sed -e 's,[^:]*:,,'`" > hail-ci-build-image
	docker tag hail-pr-builder `cat hail-ci-build-image`
	docker push `cat hail-ci-build-image`

default:
	echo Do not use this makefile to build hail, for information on how to \
	     build hail see: https://hail.is/docs/0.2/
	exit 1
