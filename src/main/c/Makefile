.PHONY: default test clean libsimdpp fullperftest test256 testSub1kB cachegrind

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
	rm -rf lib

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

fullperftest:
	${CXX} ${CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_PER_MATRIX_IN_KB=1 -o test1
	${CXX} ${CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_PER_MATRIX_IN_KB=2 -o test2
	${CXX} ${CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_PER_MATRIX_IN_KB=3 -o test3
	${CXX} ${CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_PER_MATRIX_IN_KB=4 -o test4
	${CXX} ${CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_PER_MATRIX_IN_KB=5 -o test5
	${CXX} ${CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_PER_MATRIX_IN_KB=6 -o test6
	${CXX} ${CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_PER_MATRIX_IN_KB=7 -o test7
	${CXX} ${CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_PER_MATRIX_IN_KB=8 -o test8
	${CXX} ${CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_PER_MATRIX_IN_KB=12 -o test12
	${CXX} ${CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_PER_MATRIX_IN_KB=16 -o test16
	${CXX} ${CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_PER_MATRIX_IN_KB=32 -o test32
	./test1
	./test2
	./test3
	./test4
	./test5
	./test6
	./test7
	./test8
	./test12
	./test16
	./test32

test256:
	${CXX} ${CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_PER_MATRIX_IN_KB=256 -o test256
	./test256

testSub1kB:
	${CXX} ${CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_IN_MATRIX_ROWS=1 -o test1row
	${CXX} ${CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_IN_MATRIX_ROWS=2 -o test2row
	${CXX} ${CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_IN_MATRIX_ROWS=3 -o test3row
	${CXX} ${CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_IN_MATRIX_ROWS=4 -o test4row
	./test1row
	./test2row
	./test3row
	./test4row

cachegrind:
	${CXX} ${CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_PER_MATRIX_IN_KB=1 -o test1 -DITERATIONS=2
	${CXX} ${CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_PER_MATRIX_IN_KB=2 -o test2 -DITERATIONS=2
	${CXX} ${CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_PER_MATRIX_IN_KB=3 -o test3 -DITERATIONS=2
	${CXX} ${CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_PER_MATRIX_IN_KB=4 -o test4 -DITERATIONS=2
	${CXX} ${CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_PER_MATRIX_IN_KB=5 -o test5 -DITERATIONS=2
	${CXX} ${CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_PER_MATRIX_IN_KB=6 -o test6 -DITERATIONS=2
	${CXX} ${CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_PER_MATRIX_IN_KB=7 -o test7 -DITERATIONS=2
	${CXX} ${CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_PER_MATRIX_IN_KB=8 -o test8 -DITERATIONS=2
	${CXX} ${CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_PER_MATRIX_IN_KB=12 -o test12 -DITERATIONS=2
	${CXX} ${CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_PER_MATRIX_IN_KB=16 -o test16 -DITERATIONS=2
	${CXX} ${CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_PER_MATRIX_IN_KB=32 -o test32 -DITERATIONS=2
	${CXX} ${CXXFLAGS} ibs.cpp cache-tests.cpp -DCACHE_SIZE_PER_MATRIX_IN_KB=256 -o test256 -DITERATIONS=2
	valgrind --tool=cachegrind ./test1
	valgrind --tool=cachegrind ./test2
	valgrind --tool=cachegrind ./test3
	valgrind --tool=cachegrind ./test4
	valgrind --tool=cachegrind ./test5
	valgrind --tool=cachegrind ./test6
	valgrind --tool=cachegrind ./test7
	valgrind --tool=cachegrind ./test8
	valgrind --tool=cachegrind ./test12
	valgrind --tool=cachegrind ./test16
	valgrind --tool=cachegrind ./test32
	valgrind --tool=cachegrind ./test256
