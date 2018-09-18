.PHONY: hail-ci-build-image push-hail-ci-build-image
.DEFAULT_GOAL := default

hail-ci-build-image:
	docker build . -t hail-pr-builder --rm=false -f Dockerfile.pr-builder
	echo "gcr.io/broad-ctsa/hail-pr-builder:$(shell docker images -q --no-trunc hail-pr-builder | sed -e 's,[^:]*:,,')" > hail-ci-build-image
	docker tag hail-pr-builder $(shell cat hail-ci-build-image)

push-hail-ci-build-image: hail-ci-build-image
	docker push $(shell cat hail-ci-build-image)

default:
	echo Do not use this makefile to build hail, for information on how to \
	     build hail see: https://hail.is/docs/devel/
	exit -1
