.PHONY: all test clean libsimdpp

# If you want to add a new cpp file, like foo.cpp, to the library, add foo.o to
# this list
OBJECTS := \
  ibs.o davies.o \
  NativeCodeSuite.o \
  NativeLongFunc.o \
  NativePtr.o \
  NativeStatus.o \
  Region.o

UNAME_S := $(shell uname -s)
UNAME_P := $(shell uname -p)

ifndef JAVA_HOME
  TMP :=$(shell java -XshowSettings:properties -version 2>&1 | fgrep -i java.home)
  ifneq ($(TMP),)
    JAVA_HOME := $(shell dirname $(filter-out java.home =,$(TMP)))
  endif
endif

ifeq ($(UNAME_S),Linux)
  JAVA_MD :=linux
  CXXSTD := -std=c++11
else
  JAVA_MD :=darwin
  CXXSTD := -std=c++14
endif

# Eventually we would like -std=c++17, but the CI machines have
# compilers which only go to -std=c++11

ifndef CXX
  # clang is preferred because a) it's standard on Mac, and b) it compiles faster
  CXX :=$(shell which clang)
  ifeq ($(CXX),)
    CXX :=$(shell which g++)
    ifeq ($(CXX),)
      CXX := c++
    endif
  endif
endif

# append existing flags so they override our flags
CXXFLAGS += -O3 -march=native $(CXXSTD) -Ilibsimdpp-2.0-rc2 -Wall -Werror
CXXFLAGS += -fPIC -ggdb -fno-strict-aliasing
CXXFLAGS += -I../resources/include -I$(JAVA_HOME)/include -I$(JAVA_HOME)/include/$(JAVA_MD)
LIBFLAGS += -fvisibility=default

ifeq ($(UNAME_S),Linux)
  LIBFLAGS += -rdynamic -shared
  shared_library = lib/linux-x86-64/libhail.so
  ifneq ($(filter %86,$(UNAME_P)),)
    shared_library = lib/linux-x86/libhail.so
  endif
endif
ifeq ($(UNAME_S),Darwin)
  LIBFLAGS += -dynamiclib -Wl,-undefined,dynamic_lookup
  shared_library = lib/darwin/libhail.dylib
endif

all: debug $(shared_library)

debug:
	echo "make debug"
ifndef JAVA_HOME
	echo JAVA_HOME undefined
endif
	echo "JAVA_HOME is $(JAVA_HOME)"
	echo "CXX is $(CXX)"
	-$(CXX) --version

build/functional-tests: ibs.cpp test.cpp
	mkdir -p build
	${CXX} ${CXXFLAGS} -DNUMBER_OF_GENOTYPES_PER_ROW=256 ibs.cpp test.cpp -o build/functional-tests

test: build/functional-tests
	./build/functional-tests

clean:
	-rm -rf build lib
	-rm -rf *.o
	-rm -f headers
	echo done

headers: $(shell /usr/bin/find ../resources/include -name "*.h")
	touch headers

NativeCodeSuite.o: headers
NativeLongFunc.o: headers
NativePtr.o: headers
NativeStatus.o: headers
Region.o: headers

# if your build machine cannot access this URL, download this tar.gz and place
# it in the same directory as this Makefile. If you do so, the build will
# succeed without attempting to access the Internet.
# WGET ?= wget
WGET ?= curl -L -O
libsimdpp-2.0-rc2.tar.gz:
	${WGET} https://storage.googleapis.com/hail-common/libsimdpp-2.0-rc2.tar.gz

libsimdpp-2.0-rc2: libsimdpp-2.0-rc2.tar.gz
	tar -xzf libsimdpp-2.0-rc2.tar.gz

$(shared_library): libsimdpp-2.0-rc2 ${OBJECTS}
	mkdir -p $(dir $(shared_library))
	${CXX} ${LIBFLAGS} ${LIBDIRS} ${CXXFLAGS} ${OBJECTS} -o $@ 

