.PHONY: build
build: base-stmp service-base

PROJECT := $(shell gcloud config get-value project)

BASE_LATEST = gcr.io/$(PROJECT)/base:latest
BASE_IMAGE = gcr.io/$(PROJECT)/base:$(shell docker images -q --no-trunc base:latest | sed -e 's,[^:]*:,,')

SERVICE_BASE_LATEST = gcr.io/$(PROJECT)/service-base:latest
SERVICE_BASE_IMAGE = gcr.io/$(PROJECT)/service-base:$(shell docker images -q --no-trunc service-base:latest | sed -e 's,[^:]*:,,')

HAIL_PUBLIC_LATEST = gcr.io/$(PROJECT)/hail-public:latest
HAIL_PUBLIC_IMAGE = gcr.io/$(PROJECT)/hail-public:$(shell docker images -q --no-trunc hail-public:latest | sed -e 's,[^:]*:,,')

GENETICS_PUBLIC_LATEST = gcr.io/$(PROJECT)/genetics-public:latest
GENETICS_PUBLIC_IMAGE = gcr.io/$(PROJECT)/genetics-public:$(shell docker images -q --no-trunc genetics-public:latest | sed -e 's,[^:]*:,,')

.PHONY: base
base: base-stmp

base-stmp: Dockerfile.base core-site.xml requirements.txt ../pylintrc ../setup.cfg
	-docker pull ubuntu:18.04
	-docker pull $(BASE_LATEST)
	[ "$(shell bash stat-permissions.sh Dockerfile.base)" = "644" ]
	[ "$(shell bash stat-permissions.sh core-site.xml)" = "644" ]
	[ "$(shell bash stat-permissions.sh requirements.txt)" = "644" ]
	[ "$(shell bash stat-permissions.sh ../pylintrc)" = "644" ]
	[ "$(shell bash stat-permissions.sh ../setup.cfg)" = "644" ]
	docker build -t base -f Dockerfile.base --cache-from base,$(BASE_LATEST),ubuntu:18.04 ..
	touch base-stmp

.PHONY: service-base
service-base: base-stmp
	-docker pull $(SERVICE_BASE_LATEST)
	python3 ../ci/jinja2_render.py '{"base_image":{"image":"base"}}' Dockerfile.service-base Dockerfile.service-base.out
	[ "$(shell bash stat-permissions.sh Dockerfile.service-base.out)" = "644" ]
	[ "$(shell bash stat-permissions.sh service-base-requirements.txt)" = "644" ]
	docker build -t service-base -f Dockerfile.service-base.out --cache-from service-base,$(SERVICE_BASE_LATEST),base,ubuntu:18.04 ..

.PHONY: hail-public-image
hail-public-image:
	make -C ../hail wheel
	cp ../hail/build/deploy/dist/hail-*-py3-none-any.whl hail/
	cd hail && tar -cvf wheel-container.tar hail-*-py3-none-any.whl
	-docker pull $(HAIL_PUBLIC_LATEST)
	docker build hail -f hail/Dockerfile -t hail-public \
     --cache-from hail-public,$(HAIL_PUBLIC_LATEST),ubuntu:18.04 \
     hail

.PHONY: genetics-public-image
genetics-public-image: hail-public-image
	python3 ../ci/jinja2_render.py '{"hail_public_image":{"image":"hail-public-image"}}' genetics/Dockerfile genetics/Dockerfile.out
	-docker pull $(GENETICS_PUBLIC_LATEST)
	docker build genetics -f genetics/Dockerfile.out -t genetics-public \
    --cache-from genetics-public,$(GENETICS_PUBLIC_LATEST),hail-public,$(HAIL_PUBLIC_LATEST) \
    genetics

.PHONY: push
push: build
	docker tag base $(BASE_LATEST)
	docker push $(BASE_LATEST)
	docker tag base $(BASE_IMAGE)
	docker push $(BASE_IMAGE)
	docker tag service-base $(SERVICE_BASE_LATEST)
	docker push $(SERVICE_BASE_LATEST)
	docker tag service-base $(SERVICE_BASE_IMAGE)
	docker push $(SERVICE_BASE_IMAGE)

.PHONY: deploy
deploy: push

.PHONY: clean
clean:
	rm -f base-stmp
