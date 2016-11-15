.PHONY: default

CC=cc
CCFLAGS=-O3 -march=native -g -dynamiclib -std=gnu99 -fvisibility=hidden

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
	CCFLAGS += -D LINUX
	shared_library = lib/lib/libibs.so
endif
ifeq ($(UNAME_S),Darwin)
	CCFLAGS += -D OSX
	shared_library = lib/lib/darwin/libibs.dylib
endif

default: $(shared_library)

lib/lib/darwin/lib%.dylib: %.c
	${CC} ${CCFLAGS} $? -o $@

lib/lib/lib%.so: %.c
	${CC} ${CCFLAGS} $? -o $@

clean:
	rm -r lib
