.PHONY: default test clean libsimdpp fullperftest test256 testSub1kB cachegrind

CXX ?= c++
# append existing flags so they override our flags
_CXXFLAGS = -O3 -march=native -g -std=c++11 -Ilibsimdpp-2.0-rc2 -Wall -Werror ${CXXFLAGS}
LIBFLAGS += -fvisibility=hidden

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

default: $(shared_library)

test:
	mkdir -p build
	${CXX} ${_CXXFLAGS} -DNUMBER_OF_GENOTYPES_PER_ROW=256 ibs.cpp test.cpp -o build/functional-tests
	./build/functional-tests

clean:
	rm -rf lib
	rm -rf build

# if your build machine cannot access this URL, download this tar.gz and place
# it in the same directory as this Makefile. If you do so, the build will
# succeed without attempting to access the Internet.
# WGET ?= wget
WGET ?= curl -L -O
libsimdpp-2.0-rc2.tar.gz:
	${WGET} https://storage.googleapis.com/hail-common/libsimdpp-2.0-rc2.tar.gz

libsimdpp-2.0-rc2: libsimdpp-2.0-rc2.tar.gz
	tar -xzf libsimdpp-2.0-rc2.tar.gz

libsimdpp: libsimdpp-2.0-rc2
	(cd libsimdpp-2.0-rc2 && cmake .)

lib/darwin/libibs.dylib: libsimdpp
lib/linux-x86-64/libibs.so: libsimdpp
lib/linux-x86/libibs.so: libsimdpp

lib/darwin/lib%.dylib: %.cpp
	mkdir -p lib/darwin
	${CXX} ${LIBFLAGS} ${_CXXFLAGS} $? -o $@

lib/linux-x86-64/lib%.so: %.cpp
	mkdir -p lib/linux-x86-64
	${CXX} ${LIBFLAGS} ${_CXXFLAGS} $? -o $@

lib/linux-x86/lib%.so: %.cpp
	mkdir -p lib/linux-x86
	${CXX} ${LIBFLAGS} ${_CXXFLAGS} $? -o $@
