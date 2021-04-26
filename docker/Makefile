include ../config.mk

.PHONY: build
build: base-stmp service-base hail-ubuntu

HAIL_UBUNTU_LATEST = $(DOCKER_PREFIX)/hail-ubuntu:latest
HAIL_UBUNTU_IMAGE = $(DOCKER_PREFIX)/hail-ubuntu:$(shell docker images -q --no-trunc hail-ubuntu:latest | sed -e 's,[^:]*:,,')

BASE_LATEST = $(DOCKER_PREFIX)/base:latest
BASE_IMAGE = $(DOCKER_PREFIX)/base:$(shell docker images -q --no-trunc base:latest | sed -e 's,[^:]*:,,')

SERVICE_BASE_LATEST = $(DOCKER_PREFIX)/service-base:latest
SERVICE_BASE_IMAGE = $(DOCKER_PREFIX)/service-base:$(shell docker images -q --no-trunc service-base:latest | sed -e 's,[^:]*:,,')

SERVICE_JAVA_RUN_BASE_LATEST = $(DOCKER_PREFIX)/service-java-run-base:latest
SERVICE_JAVA_RUN_BASE_IMAGE = $(DOCKER_PREFIX)/service-java-run-base:$(shell docker images -q --no-trunc service-java-run-base:latest | sed -e 's,[^:]*:,,')

HAIL_PUBLIC_LATEST = $(DOCKER_PREFIX)/hail-public:latest
HAIL_PUBLIC_IMAGE = $(DOCKER_PREFIX)/hail-public:$(shell docker images -q --no-trunc hail-public:latest | sed -e 's,[^:]*:,,')

GENETICS_PUBLIC_LATEST = $(DOCKER_PREFIX)/genetics-public:latest
GENETICS_PUBLIC_IMAGE = $(DOCKER_PREFIX)/genetics-public:$(shell docker images -q --no-trunc genetics-public:latest | sed -e 's,[^:]*:,,')

.PHONY: hail-ubuntu
hail-ubuntu: hail-ubuntu-stmp

hail-ubuntu-stmp: hail-ubuntu/Dockerfile hail-ubuntu/hail-pip-install hail-ubuntu/pip.conf hail-ubuntu/hail-apt-get-install
	-docker pull $(DOCKER_PREFIX)/ubuntu:bionic-20200921
	-docker pull $(HAIL_UBUNTU_LATEST)
	python3 ../ci/jinja2_render.py '{"global":{"docker_prefix":"$(DOCKER_PREFIX)"}}' hail-ubuntu/Dockerfile hail-ubuntu/Dockerfile.out
	docker build -t hail-ubuntu -f hail-ubuntu/Dockerfile.out --cache-from hail-ubuntu,$(HAIL_UBUNTU_LATEST),ubuntu:bionic-20200921 hail-ubuntu
	touch hail-ubuntu-stmp

.PHONY: base
base: base-stmp

base-stmp: hail-ubuntu-stmp Dockerfile.base core-site.xml requirements.txt ../pylintrc ../setup.cfg
	-docker pull $(BASE_LATEST)
	python3 ../ci/jinja2_render.py '{"hail_ubuntu_image":{"image":"hail-ubuntu"}}' Dockerfile.base Dockerfile.base.out
	[ "$(shell bash stat-permissions.sh Dockerfile.base)" = "644" ]
	[ "$(shell bash stat-permissions.sh core-site.xml)" = "644" ]
	[ "$(shell bash stat-permissions.sh requirements.txt)" = "644" ]
	[ "$(shell bash stat-permissions.sh ../pylintrc)" = "644" ]
	[ "$(shell bash stat-permissions.sh ../setup.cfg)" = "644" ]
	docker build -t base -f Dockerfile.base.out --cache-from base,$(BASE_LATEST),hail-ubuntu ..
	touch base-stmp

.PHONY: service-base
service-base: base-stmp
	-docker pull $(SERVICE_BASE_LATEST)
	$(MAKE) -C ../hail python/hailtop/hail_version
	cp ../hail/python/hailtop/hail_version ../hail_version
	python3 ../ci/jinja2_render.py '{"base_image":{"image":"base"}}' Dockerfile.service-base Dockerfile.service-base.out
	[ "$(shell bash stat-permissions.sh Dockerfile.service-base.out)" = "644" ]
	[ "$(shell bash stat-permissions.sh service-base-requirements.txt)" = "644" ]
	docker build -t service-base -f Dockerfile.service-base.out --cache-from service-base,$(SERVICE_BASE_LATEST),base,hail-ubuntu ..

.PHONY: service-java-run-base
service-java-run-base: hail-ubuntu-stmp
	-docker pull $(SERVICE_JAVA_RUN_BASE_LATEST)
	$(MAKE) -C ../hail python/hailtop/hail_version
	cp ../hail/python/hailtop/hail_version ../hail_version
	python3 ../ci/jinja2_render.py '{"hail_ubuntu_image":{"image":"hail-ubuntu"}}' Dockerfile.service-java-run-base Dockerfile.service-java-run-base.out
	[ "$(shell bash stat-permissions.sh Dockerfile.service-java-run-base.out)" = "644" ]
	docker build -t service-java-run-base -f Dockerfile.service-java-run-base.out --cache-from service-java-run-base,$(SERVICE_JAVA_RUN_BASE_LATEST),base,hail-ubuntu ..

.PHONY: push
push: build
	docker tag hail-ubuntu $(HAIL_UBUNTU_LATEST)
	docker push $(HAIL_UBUNTU_LATEST)
	docker tag hail-ubuntu $(HAIL_UBUNTU_IMAGE)
	docker push $(HAIL_UBUNTU_IMAGE)
	docker tag base $(BASE_LATEST)
	docker push $(BASE_LATEST)
	docker tag base $(BASE_IMAGE)
	docker push $(BASE_IMAGE)
	docker tag service-base $(SERVICE_BASE_LATEST)
	docker push $(SERVICE_BASE_LATEST)
	docker tag service-base $(SERVICE_BASE_IMAGE)
	docker push $(SERVICE_BASE_IMAGE)
	docker tag service-java-run-base $(SERVICE_JAVA_RUN_BASE_LATEST)
	docker push $(SERVICE_JAVA_RUN_BASE_LATEST)
	docker tag service-java-run-base $(SERVICE_JAVA_RUN_BASE_IMAGE)
	docker push $(SERVICE_JAVA_RUN_BASE_IMAGE)

.PHONY: deploy
deploy: push

.PHONY: clean
clean:
	rm -f base-stmp hail-ubuntu-stmp

## Public Images ###############################################################

publish-public-images:
	bash publish-public-images.sh $(PROJECT)
