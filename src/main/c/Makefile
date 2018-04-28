.PHONY: default test clean libsimdpp

# If you want to add a new cpp file, like foo.cpp, to the library, add foo.o to
# this list
OBJECTS := \
  ibs.o davies.o \
  NativeModule.o NativeObj.o NativePtr.o \
  NativeStatus.o NativeLongFunc.o NativeCodeSuite.o

UNAME_S := $(shell uname -s)
UNAME_P := $(shell uname -p)

# Compile with new version of clang if it is installed
ifeq ($(UNAME_S),Linux)
  CLANG_VERSION := 5.0
  CLANG_HOME := /usr/lib/llvm-$(CLANG_VERSION)
  JAVA_HOME := /usr/lib/jvm/default-java
  JAVA_MD := linux
else
  CLANG_VERSION := 6.0.0
  CLANG_HOME := /usr/local/llvm-$(CLANG_VERSION)
  JAVA_HOME := /Library/Java/JavaVirtualMachines/jdk1.8.0_162.jdk/Contents/Home
  JAVA_MD := darwin
endif

ifndef JAVA_HOME
  JAVA_HOME := JAVA_HOME_undefined
endif

ifeq ($(shell [ -d $(CLANG_HOME) ] && echo found),found)
  CXX := $(CLANG_HOME)/bin/clang
  LIBDIRS := -L$(CLANG_HOME)/lib
else
  CXX ?= c++
  LIBDIRS :=
endif

ifeq ($(UNAME_S),Darwin)
  # CXX := c++
endif

# Eventually we would like -std=c++17, but the CI machines have
# compilers which only go to -std=c++11

# append existing flags so they override our flags
CXXFLAGS += -O3 -march=native -g -std=c++11 -Ilibsimdpp-2.0-rc2 -Wall -Werror
ifeq ($(UNAME_S),Linux)
#CXXFLAGS += --no-strict-aliasing
endif
CXXFLAGS += -fPIC -ggdb -fno-strict-aliasing
CXXFLAGS += -I$(HAIL_HOME)/include -I$(JAVA_HOME)/include -I$(JAVA_HOME)/include/$(JAVA_MD)
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

all: $(shared_library)

debug:
	echo "make debug"
	echo "CXX is $(CXX)"

build/functional-tests: ibs.cpp test.cpp
	mkdir -p build
	${CXX} ${CXXFLAGS} -DNUMBER_OF_GENOTYPES_PER_ROW=256 ibs.cpp test.cpp -o build/functional-tests

test: build/functional-tests
	./build/functional-tests

clean:
	@echo CXX $(CXX)
	rm -rf lib
	rm -rf build
	rm -rf *.o
	echo done



headers: $(shell /usr/bin/find ../../../include -name "*.h")
	touch headers

NativeModule.o: headers
NativeObj.o: headers
NativePtr.o: headers
NativeStatus.o: headers
NativeLongFunc.o: headers
NativeCodeSuite.o: headers

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
	mkdir -p $(basename $(shared_library))
	${CXX} ${LIBFLAGS} ${LIBDIRS} ${CXXFLAGS} ${OBJECTS} -o $@

