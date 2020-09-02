.DEFAULT_GOAL := default

default:
	echo Do not use this makefile to build hail, for information on how to \
	     build hail see: https://hail.is/docs/0.2/
	exit 1

.PHONY: check-all
check-all: check-hail check-services check-benchmark-service

.PHONY: check-hail
check-hail:
	make -C hail/python check

.PHONY: check-services
check-services: check-auth check-batch check-ci check-gear check-notebook \
  check-query check-router-resolver check-scorecard check-web-common

.PHONY: check-auth
check-auth:
	make -C auth check

.PHONY: check-batch
check-batch:
	make -C batch check

.PHONY: check-benchmark-service
check-benchmark-service:
	make -C benchmark-service check

.PHONY: check-ci
check-ci:
	make -C ci check

.PHONY: check-gear
check-gear:
	make -C gear check

.PHONY: check-notebook
check-notebook:
	make -C notebook check

.PHONY: check-query
check-query:
	make -C query check

.PHONY: check-router-resolver
check-router-resolver:
	make -C router-resolver check

.PHONY: check-scorecard
check-scorecard:
	make -C scorecard check

.PHONY: check-web-common
check-web-common:
	make -C web_common check
