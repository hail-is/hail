.PHONY: default libsimdpp

CXX=c++
CXXFLAGS=-O3 -march=native -g -std=c++11 -Ilibsimdpp-2.0-rc2 -DSIMDPP_ARCH_X86_AVX2
LIBFLAGS=-fvisibility=hidden

UNAME_S := $(shell uname -s)
UNAME_P := $(shell uname -p)
ifeq ($(UNAME_S),Linux)
       LIBFLAGS += -rdynamic -shared -fPIC -ggdb
       shared_library = lib/linux-x86-64/libibs.so
       ifneq ($(filter %86,$(UNAME_P)),)
               shared_library = lib/linux-x86/libibs.so
       endif
endif
ifeq ($(UNAME_S),Darwin)
       LIBFLAGS += -dynamiclib
       shared_library = lib/darwin/libibs.dylib
endif

default: libsimdpp $(shared_library)

test:
	${CXX} ${CXXFLAGS} ibs.cpp test.cpp -o functional-tests
	./functional-tests

clean:
	rm -r lib

libsimdpp:
	(cd libsimdpp-2.0-rc2 && cmake .)

lib/darwin/lib%.dylib: %.cpp
	mkdir -p lib/darwin
	${CXX} ${LIBFLAGS} ${CXXFLAGS} $? -o $@

lib/linux-x86-64/lib%.so: %.cpp
	mkdir -p lib/linux-x86-64
	${CXX} ${LIBFLAGS} ${CXXFLAGS} $? -o $@

lib/linux-x86/lib%.so: %.cpp
	mkdir -p lib/linux-x86
	${CXX} ${LIBFLAGS} ${CXXFLAGS} $? -o $@
