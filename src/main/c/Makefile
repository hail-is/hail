.PHONY: default

CC=cc
CCFLAGS=-O3 -march=native -g -std=gnu99 -fvisibility=hidden

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
	CCFLAGS += -rdynamic -shared -fPIC -ggdb
	ifeq ($(UNAME_P),x86_64)
		shared_library = lib/lib/linux-x86-64/libibs.so
	endif
	ifneq ($(filter %86,$(UNAME_P)),)
		shared_library = lib/lib/linux-x86/libibs.so
	endif
endif
ifeq ($(UNAME_S),Darwin)
	CCFLAGS += -dynamiclib
	shared_library = lib/lib/darwin/libibs.dylib
endif

default: $(shared_library)

lib/lib/darwin/lib%.dylib: %.c
	mkdir -p lib/lib/darwin
	${CC} ${CCFLAGS} $? -o $@

lib/lib/linux-x86-64/lib%.so: %.c
	mkdir -p lib/lib/linux-x86-64
	${CC} ${CCFLAGS} $? -o $@

lib/lib/linux-x86/lib%.so: %.c
	mkdir -p lib/lib/linux-x86
	${CC} ${CCFLAGS} $? -o $@

clean:
	rm -r lib
