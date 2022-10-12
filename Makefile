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
  check-notebook check-web-common check-website

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

.PHONY: check-web-common
check-web-common:
	$(MAKE) -C web_common check

.PHONY: check-website
check-website:
	$(MAKE) -C website check

.PHONY: check-pip-dependencies
check-pip-dependencies:
	./check_pip_requirements.sh hail/python/requirements.txt hail/python/pinned-requirements.txt
	./check_pip_requirements.sh hail/python/dev/requirements.txt hail/python/dev/pinned-requirements.txt
	./check_pip_requirements.sh docker/requirements.txt docker/linux-pinned-requirements.txt
