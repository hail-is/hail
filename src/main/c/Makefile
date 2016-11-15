.PHONY: default

CC=cc
CCFLAGS=-O3 -march=native -g -std=gnu99 -fvisibility=hidden

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
	CCFLAGS += -rdynamic -shared -fPIC -ggdb
	shared_library = lib/lib/libibs.so
endif
ifeq ($(UNAME_S),Darwin)
	CCFLAGS += -dynamiclib
	shared_library = lib/lib/darwin/libibs.dylib
endif

default: $(shared_library)

lib/lib/darwin/lib%.dylib: %.c
	${CC} ${CCFLAGS} $? -o $@

lib/lib/lib%.so: %.c
	${CC} ${CCFLAGS} $? -o $@

clean:
	rm -r lib
