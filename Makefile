.PHONY: hail-ci-build-image push-hail-ci-build-image
.DEFAULT_GOAL := default

hail-ci-build-image:
	-docker pull $(shell cat hail-ci-build-image) # pull the image at least once so it can be used as a cache source
	docker build . -t hail-pr-builder -f Dockerfile.pr-builder

push-hail-ci-build-image: hail-ci-build-image
	echo "gcr.io/broad-ctsa/hail-pr-builder:`docker images -q --no-trunc hail-pr-builder | sed -e 's,[^:]*:,,'`" > hail-ci-build-image
	docker tag hail-pr-builder `cat hail-ci-build-image`
	docker push `cat hail-ci-build-image`

default:
	echo Do not use this makefile to build hail, for information on how to \
	     build hail see: https://hail.is/docs/devel/
	exit 1
