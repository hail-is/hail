.PHONY: default test clean libsimdpp fullperftest test256 testSub1kB cachegrind

CXX ?= c++
_CXXFLAGS += -O3 -march=native -g -std=c++11 -Ilibsimdpp-2.0-rc2 ${CXXFLAGS}
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

default: libsimdpp $(shared_library)

test:
	${CXX} ${_CXXFLAGS} ibs.cpp test.cpp -o build/functional-tests
	./build/functional-tests

clean:
	rm -rf lib
	rm -rf build

libsimdpp:
	(cd libsimdpp-2.0-rc2 && cmake .)

lib/darwin/lib%.dylib: %.cpp
	mkdir -p lib/darwin
	${CXX} ${LIBFLAGS} ${_CXXFLAGS} $? -o $@

lib/linux-x86-64/lib%.so: %.cpp
	mkdir -p lib/linux-x86-64
	${CXX} ${LIBFLAGS} ${_CXXFLAGS} $? -o $@

lib/linux-x86/lib%.so: %.cpp
	mkdir -p lib/linux-x86
	${CXX} ${LIBFLAGS} ${_CXXFLAGS} $? -o $@

fullperftest:
	mkdir -p build
	${CXX} ${_CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_PER_MATRIX_IN_KB=1 -o build/test1
	${CXX} ${_CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_PER_MATRIX_IN_KB=2 -o build/test2
	${CXX} ${_CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_PER_MATRIX_IN_KB=3 -o build/test3
	${CXX} ${_CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_PER_MATRIX_IN_KB=4 -o build/test4
	${CXX} ${_CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_PER_MATRIX_IN_KB=5 -o build/test5
	${CXX} ${_CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_PER_MATRIX_IN_KB=6 -o build/test6
	${CXX} ${_CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_PER_MATRIX_IN_KB=7 -o build/test7
	${CXX} ${_CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_PER_MATRIX_IN_KB=8 -o build/test8
	${CXX} ${_CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_PER_MATRIX_IN_KB=12 -o build/test12
	${CXX} ${_CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_PER_MATRIX_IN_KB=16 -o build/test16
	${CXX} ${_CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_PER_MATRIX_IN_KB=32 -o build/test32
	./build/test1
	./build/test2
	./build/test3
	./build/test4
	./build/test5
	./build/test6
	./build/test7
	./build/test8
	./build/test12
	./build/test16
	./build/test32

test256:
	mkdir -p build
	${CXX} ${_CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_PER_MATRIX_IN_KB=256 -o build/test256
	./build/test256

testSub1kB:
	mkdir -p build
	${CXX} ${_CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_IN_MATRIX_ROWS=1 -o build/test1row
	${CXX} ${_CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_IN_MATRIX_ROWS=2 -o build/test2row
	${CXX} ${_CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_IN_MATRIX_ROWS=3 -o build/test3row
	${CXX} ${_CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_IN_MATRIX_ROWS=4 -o build/test4row
	./build/test1row
	./build/test2row
	./build/test3row
	./build/test4row

cachegrind:
	mkdir -p build
	${CXX} ${_CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_PER_MATRIX_IN_KB=1 -o build/test1 -DITERATIONS=2
	${CXX} ${_CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_PER_MATRIX_IN_KB=2 -o build/test2 -DITERATIONS=2
	${CXX} ${_CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_PER_MATRIX_IN_KB=3 -o build/test3 -DITERATIONS=2
	${CXX} ${_CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_PER_MATRIX_IN_KB=4 -o build/test4 -DITERATIONS=2
	${CXX} ${_CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_PER_MATRIX_IN_KB=5 -o build/test5 -DITERATIONS=2
	${CXX} ${_CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_PER_MATRIX_IN_KB=6 -o build/test6 -DITERATIONS=2
	${CXX} ${_CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_PER_MATRIX_IN_KB=7 -o build/test7 -DITERATIONS=2
	${CXX} ${_CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_PER_MATRIX_IN_KB=8 -o build/test8 -DITERATIONS=2
	${CXX} ${_CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_PER_MATRIX_IN_KB=12 -o build/test12 -DITERATIONS=2
	${CXX} ${_CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_PER_MATRIX_IN_KB=16 -o build/test16 -DITERATIONS=2
	${CXX} ${_CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_PER_MATRIX_IN_KB=32 -o build/test32 -DITERATIONS=2
	${CXX} ${_CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_PER_MATRIX_IN_KB=256 -o build/test256 -DITERATIONS=2
	valgrind --tool=cachegrind ./build/test1
	valgrind --tool=cachegrind ./build/test2
	valgrind --tool=cachegrind ./build/test3
	valgrind --tool=cachegrind ./build/test4
	valgrind --tool=cachegrind ./build/test5
	valgrind --tool=cachegrind ./build/test6
	valgrind --tool=cachegrind ./build/test7
	valgrind --tool=cachegrind ./build/test8
	valgrind --tool=cachegrind ./build/test12
	valgrind --tool=cachegrind ./build/test16
	valgrind --tool=cachegrind ./build/test32
	valgrind --tool=cachegrind ./build/test256
