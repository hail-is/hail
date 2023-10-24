REVISION := $(shell git rev-parse HEAD)
ifndef REVISION
$(error "git rev-parse HEAD" failed to produce output)
endif

SHORT_REVISION := $(shell git rev-parse --short=12 HEAD)
ifndef SHORT_REVISION
$(error "git rev-parse --short=12 HEAD" failed to produce output)
endif

BRANCH := $(shell git rev-parse --abbrev-ref HEAD)
ifndef BRANCH
$(error "git rev-parse --abbrev-ref HEAD" failed to produce output)
endif

SCALA_VERSION ?= 2.12.15
SPARK_VERSION ?= 3.3.0
HAIL_MAJOR_MINOR_VERSION := 0.2
HAIL_PATCH_VERSION := 124
HAIL_PIP_VERSION := $(HAIL_MAJOR_MINOR_VERSION).$(HAIL_PATCH_VERSION)
HAIL_VERSION := $(HAIL_PIP_VERSION)-$(SHORT_REVISION)
ELASTIC_MAJOR_VERSION ?= 7
