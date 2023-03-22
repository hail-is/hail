.DEFAULT_GOAL := default

default:
	@echo Do not use this makefile to build hail, for information on how to \
	     build hail see: https://hail.is/docs/0.2/
	@false

.PHONY: check-all
check-all: check-hail check-services

.PHONY: check-hail
check-hail:
	$(MAKE) -C hail/python check

.PHONY: check-services
check-services: check-auth check-batch check-ci check-gear check-memory \
  check-notebook check-monitoring check-web-common check-website

.PHONY: check-auth
check-auth:
	$(MAKE) -C auth check

.PHONY: check-batch
check-batch:
	$(MAKE) -C batch check

.PHONY: check-ci
check-ci:
	$(MAKE) -C ci check

.PHONY: check-gear
check-gear:
	$(MAKE) -C gear check

.PHONY: check-memory
check-memory:
	$(MAKE) -C memory check

.PHONY: check-notebook
check-notebook:
	$(MAKE) -C notebook check

.PHONY: check-monitoring
	$(MAKE) -C monitoring check

.PHONY: check-web-common
check-web-common:
	$(MAKE) -C web_common check

.PHONY: check-website
check-website:
	$(MAKE) -C website check

.PHONY: check-pip-requirements
check-pip-requirements:
	./check_pip_requirements.sh \
		hail/python/hailtop \
		hail/python \
		hail/python/dev \
		gear \
		web_common \
		auth \
		batch \
		ci \
		memory

.PHONY: install-dev-requirements
install-dev-requirements:
	python3 -m pip install \
		-r hail/python/pinned-requirements.txt \
		-r hail/python/dev/pinned-requirements.txt \
		-r gear/pinned-requirements.txt \
		-r web_common/pinned-requirements.txt \
		-r auth/pinned-requirements.txt \
		-r batch/pinned-requirements.txt \
		-r ci/pinned-requirements.txt \
		-r memory/pinned-requirements.txt \

hail/python/hailtop/pinned-requirements.txt: hail/python/hailtop/requirements.txt
	./generate-linux-pip-lockfile.sh hail/python/hailtop

hail/python/pinned-requirements.txt: hail/python/requirements.txt hail/python/hailtop/pinned-requirements.txt
	./generate-linux-pip-lockfile.sh hail/python

hail/python/dev/pinned-requirements.txt: hail/python/dev/requirements.txt hail/python/pinned-requirements.txt
	./generate-linux-pip-lockfile.sh hail/python/dev

gear/pinned-requirements.txt: hail/python/hailtop/pinned-requirements.txt gear/requirements.txt
	./generate-linux-pip-lockfile.sh gear

web_common/pinned-requirements.txt: gear/pinned-requirements.txt web_common/requirements.txt
	./generate-linux-pip-lockfile.sh web_common

auth/pinned-requirements.txt: web_common/pinned-requirements.txt auth/requirements.txt
	./generate-linux-pip-lockfile.sh auth

batch/pinned-requirements.txt: web_common/pinned-requirements.txt batch/requirements.txt
	./generate-linux-pip-lockfile.sh batch

ci/pinned-requirements.txt: web_common/pinned-requirements.txt ci/requirements.txt
	./generate-linux-pip-lockfile.sh ci

memory/pinned-requirements.txt: gear/pinned-requirements.txt memory/requirements.txt
	./generate-linux-pip-lockfile.sh memory

.PHONY: generate-pip-lockfiles
generate-pip-lockfiles: hail/python/hailtop/pinned-requirements.txt
generate-pip-lockfiles: hail/python/pinned-requirements.txt
generate-pip-lockfiles: hail/python/dev/pinned-requirements.txt
generate-pip-lockfiles: gear/pinned-requirements.txt
generate-pip-lockfiles: web_common/pinned-requirements.txt
generate-pip-lockfiles: auth/pinned-requirements.txt
generate-pip-lockfiles: batch/pinned-requirements.txt
generate-pip-lockfiles: ci/pinned-requirements.txt
generate-pip-lockfiles: memory/pinned-requirements.txt
