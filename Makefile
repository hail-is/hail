.PHONY: hail-ci-build-image

BUILD_IMAGE_SHORT_NAME = cloud-tools-pr-builder

latest-hail-ci-build-image:
	cd pr-builder && docker build . -t ${BUILD_IMAGE_SHORT_NAME}

hail-ci-build-image: HASH = $(shell docker images -q --no-trunc ${BUILD_IMAGE_SHORT_NAME} | head -n 1 | sed -e 's,[^:]*:,,')
hail-ci-build-image: latest-hail-ci-build-image
	docker tag ${BUILD_IMAGE_SHORT_NAME} ${BUILD_IMAGE_SHORT_NAME}:${HASH}

push-hail-ci-build-image: HASH = $(shell docker images -q --no-trunc ${BUILD_IMAGE_SHORT_NAME} | head -n 1 | sed -e 's,[^:]*:,,')
push-hail-ci-build-image: hail-ci-build-image
	docker tag ${BUILD_IMAGE_SHORT_NAME}:${HASH} gcr.io/broad-ctsa/${BUILD_IMAGE_SHORT_NAME}:${HASH}
	docker push gcr.io/broad-ctsa/${BUILD_IMAGE_SHORT_NAME}:${HASH}
	echo gcr.io/broad-ctsa/${BUILD_IMAGE_SHORT_NAME}:${HASH} > hail-ci-build-image

deploy: push-hail-ci-build-image
	rm -f dist/*
	python setup.py bdist_wheel
	twine upload dist/* -u ${PYPI_USERNAME} -p ${PYPI_PASSWORD}
