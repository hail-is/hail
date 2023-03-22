include ../config.mk

.PHONY: build
build: base service-base hail-ubuntu

HAIL_UBUNTU_IMAGE := $(DOCKER_PREFIX)/hail-ubuntu:$(TOKEN)
BASE_IMAGE := $(DOCKER_PREFIX)/base:$(TOKEN)
SERVICE_BASE_IMAGE := $(DOCKER_PREFIX)/service-base:$(TOKEN)
PRIVATE_REPO_HAILGENETICS_HAIL_IMAGE := $(DOCKER_PREFIX)/hailgenetics/hail:$(TOKEN)

.PHONY: hail-ubuntu hail-ubuntu-image-ref
hail-ubuntu: hail-ubuntu-image-ref
hail-ubuntu-image-ref: hail-ubuntu/Dockerfile hail-ubuntu/hail-pip-install hail-ubuntu/pip.conf hail-ubuntu/hail-apt-get-install
	python3 ../ci/jinja2_render.py '{"global":{"docker_prefix":"$(DOCKER_PREFIX)"}}' hail-ubuntu/Dockerfile hail-ubuntu/Dockerfile.out
	../docker-build.sh hail-ubuntu Dockerfile.out $(HAIL_UBUNTU_IMAGE)
	echo $(HAIL_UBUNTU_IMAGE) > $@

.PHONY: base base-image-ref
base: base-image-ref
base-image-ref: hail-ubuntu-image-ref Dockerfile.base core-site.xml requirements.txt ../pylintrc ../setup.cfg
	python3 ../ci/jinja2_render.py '{"hail_ubuntu_image":{"image":"'$$(cat hail-ubuntu-image-ref)'"}}' Dockerfile.base Dockerfile.base.out
	../docker-build.sh .. docker/Dockerfile.base.out $(BASE_IMAGE)
	echo $(BASE_IMAGE) > $@

.PHONY: hail_version
hail_version:
	$(MAKE) -C ../hail python/hailtop/hail_version
	cp ../hail/python/hailtop/hail_version ../hail_version

.PHONY: service-base service-base-image-ref
service-base: service-base-image-ref
service-base-image-ref: base hail_version
	python3 ../ci/jinja2_render.py '{"base_image":{"image":"'$$(cat base-image-ref)'"}}' Dockerfile.service-base Dockerfile.service-base.out
	../docker-build.sh .. docker/Dockerfile.service-base.out $(SERVICE_BASE_IMAGE)
	echo $(SERVICE_BASE_IMAGE) > $@

.PHONY: private-repo-hailgenetics-hail private-repo-hailgenetics-hail-image-ref
private-repo-hailgenetics-hail: private-repo-hailgenetics-hail-image-ref
private-repo-hailgenetics-hail-image-ref: hail-ubuntu-image-ref hailgenetics/hail/Dockerfile
	$(MAKE) -C ../hail wheel generate-pip-lockfile
	cp ../hail/build/deploy/dist/hail-$$(cat ../hail/python/hail/hail_pip_version)-py3-none-any.whl .
	tar -cvf ../wheel-container.tar hail-$$(cat ../hail/python/hail/hail_pip_version)-py3-none-any.whl
	rm hail-$$(cat ../hail/python/hail/hail_pip_version)-py3-none-any.whl
	python3 ../ci/jinja2_render.py '{"hail_ubuntu_image":{"image":"'$$(cat hail-ubuntu-image-ref)'"}}' hailgenetics/hail/Dockerfile hailgenetics/hail/Dockerfile.out
	../docker-build.sh .. docker/hailgenetics/hail/Dockerfile.out $(PRIVATE_REPO_HAILGENETICS_HAIL_IMAGE)
	rm ../wheel-container.tar
	echo $(PRIVATE_REPO_HAILGENETICS_HAIL_IMAGE) > $@

.PHONY: deploy
deploy: build

.PHONY: generate-pip-lockfile
generate-pip-lockfile:
	HAIL_HAIL_DIR=$(HAIL) ../generate_pip_lockfile.sh requirements.txt linux-pinned-requirements.txt

.PHONY: clean
clean:
	rm -f base hail-ubuntu-image-ref base service-base
