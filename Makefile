.PHONY: hail-ci-build-image push-hail-ci-build-image clean
.DEFAULT_GOAL := default

PR_BUILDER_FILES := \
	build.gradle \
	gradle \
	gradlew \
	python/hail/dev-environment.yml \
	settings.gradle

clean:
	[ -e pr-builder ] && cd pr-builder && rm -rf $(notdir ${PR_BUILDER_FILES})

hail-ci-build-image: GIT_SHA = $(shell git rev-parse HEAD)
hail-ci-build-image:
	mkdir -p pr-builder
	cp -R ${PR_BUILDER_FILES} pr-builder
	cd pr-builder && docker build . -t hail-pr-builder/0.1:${GIT_SHA}

push-hail-ci-build-image: GIT_SHA = $(shell git rev-parse HEAD)
push-hail-ci-build-image: hail-ci-build-image
	docker tag hail-pr-builder/0.1:${GIT_SHA} gcr.io/broad-ctsa/hail-pr-builder/0.1:${GIT_SHA}
	docker push gcr.io/broad-ctsa/hail-pr-builder/0.1
	echo gcr.io/broad-ctsa/hail-pr-builder/0.1:${GIT_SHA} > hail-ci-build-image

default:
	echo Do not use this makefile to build hail, for information on how to \
	     build hail see: https://hail.is/docs/devel/
	exit -1
