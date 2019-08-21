#FIXME check makeflags for jobs arg
ifndef PARALLELISM
OS ?= $(shell uname -s)
ifeq ($(OS),Linux)
PARALLELISM := $(shell grep -c ^processor /proc/cpuinfo)
else ifeq ($(OS),Darwin)
PARALLELISM := $(shell sysctl -n hw.logicalcpu)
else
PARALLELISM := 1
endif
endif

MAKEFLAGS += --jobs $(PARALLELISM) --load-average $(PARALLELISM)
