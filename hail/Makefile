.PHONY: hail-ci-build-image push-hail-ci-build-image
.DEFAULT_GOAL := default

hail-ci-build-image: GIT_SHA = $(shell git rev-parse HEAD) 
hail-ci-build-image:
	# pull the image at least once so it can be used as a cache source
	docker pull $(shell cat ../hail-ci-build-image)
	docker build . -t hail-pr-builder:${GIT_SHA} -f Dockerfile.pr-builder --cache-from $(shell cat ../hail-ci-build-image)

push-hail-ci-build-image: GIT_SHA = $(shell git rev-parse HEAD)
push-hail-ci-build-image: hail-ci-build-image
	docker tag hail-pr-builder:${GIT_SHA} gcr.io/broad-ctsa/hail-pr-builder:${GIT_SHA}
	docker push gcr.io/broad-ctsa/hail-pr-builder
	echo gcr.io/broad-ctsa/hail-pr-builder:${GIT_SHA} > ../hail-ci-build-image

default:
	echo Do not use this makefile to build hail, for information on how to \
	     build hail see: https://hail.is/docs/devel/
	exit -1
