.PHONY hail-ci-build-image push-hail-ci-build-image

hail-ci-build-image:
	docker build . -t hail-pr-builder -f Dockerfile.pr-builder

push-hail-ci-build-image:
	docker tag hail-pr-builder gcr.io/broad-ctsa/hail-pr-builder
	docker push gcr.io/broad-ctsa/hail-pr-builder
