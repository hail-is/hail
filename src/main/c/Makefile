# This makefile calls itself recursively to build the libraries with
# different compiler options for different ABI versions:
#   -fabi-version=2 was the default from g++-3.4 to g++-4.9
#   -fabi-version=9 was the default in g++-5.x

ifndef SUBMAKE_NAME

.PHONY: top all clean debug test

top: all

all clean debug test:
	$(MAKE) SUBMAKE_NAME=abi_v2 SUBMAKE_CXXFLAGS=-fabi-version=2 $@
	$(MAKE) SUBMAKE_NAME=abi_v9 SUBMAKE_CXXFLAGS=-fabi-version=9 $@

else

.PHONY: all clean debug test libsimdpp

BUILD := build_$(SUBMAKE_NAME)
BUILD_FORCE := $(shell mkdir -p $(BUILD))

# If you want to add a new cpp file, like foo.cpp, to the library, add foo.o to
# this list
OBJECTS := \
  davies.o \
  ibs.o \
  NativeCodeSuite.o \
  NativeLongFunc.o \
  NativeModule.o \
  NativePtr.o \
  NativeStatus.o \
  Region.o

UNAME_S := $(shell uname -s)
UNAME_P := $(shell uname -p)

BUILD_OBJECTS := $(OBJECTS:%=$(BUILD)/%)

$(BUILD)/%.o: %.cpp
	$(CXX) -o $@ $(CXXFLAGS) -c $<

ifndef JAVA_HOME
  TMP :=$(shell java -XshowSettings:properties -version 2>&1 | fgrep -i java.home)
  ifneq ($(TMP),)
    JAVA_HOME := $(shell dirname $(filter-out java.home =,$(TMP)))
  endif
endif

ifeq ($(UNAME_S),Linux)
  JAVA_MD :=linux
else
  JAVA_MD :=darwin
endif

# Currently source code for libhail only uses features up to C++11. The
# intention is to support C++17 for dynamic-compiled code eventually, but 
# to maximize compatibility with different compilers/platforms we don't
# require that here.

CXXSTD := -std=c++11

# Check for any inherited CXXFLAGS which could interfere with
# ABI compatibility.  Such flags will cause a warning, then will be
# ignored.

BADFLAGS := \
  -fabi-version=% -f%-struct-return -fshort-enums -fshort-wchar -fpie -fPIE -ffixed-% \
  -fcall-used-% -fcall-saved-% -fpack-struct% -f%leading-underscore -f%trampolines -fvisibility=% \
  -f%strict-volatile-bitfields 

WARNFLAGS :=$(filter $(BADFLAGS),$(CXXFLAGS))
ifneq ($(WARNFLAGS),)
  $(warning WARNING: ignored CXXFLAGS options affecting binary compatibility: $(WARNFLAGS))
  CXXFLAGS := $(filter-out $(WARNFLAGS),$(CXXFLAGS))
endif

# If no inherited "-march=%", then use "-march=sandybridge" for ISA compatibility
# with MacBook Pro's since 2011 (also the earliest cpu with AVX).

ifeq ($(filter -march=%,$(CXXFLAGS)),)
  CXXFLAGS += -march=sandybridge
endif

# Append to any inherited flags which survived filtering
CXXFLAGS += -O3 $(CXXSTD) -Ilibsimdpp-2.0-rc2 -Wall -Werror
CXXFLAGS += -fPIC -ggdb -fno-strict-aliasing
CXXFLAGS += -I../resources/include -I$(JAVA_HOME)/include -I$(JAVA_HOME)/include/$(JAVA_MD)
LIBFLAGS += -fvisibility=default
PREBUILT := ../../../prebuilt

ifeq ($(UNAME_S),Linux)
  LIBFLAGS += -rdynamic -shared
  LIBBOOT := lib/linux-x86-64/libboot.so
  LIBHAIL := lib/linux-x86-64/libhail_$(SUBMAKE_NAME).so
  ifneq ($(filter %86,$(UNAME_P)),)
    LIBBOOT := lib/linux-x86/libboot.so
    LIBHAIL := lib/linux-x86/libhail_$(SUBMAKE_NAME).so
  endif
endif
ifeq ($(UNAME_S),Darwin)
  LIBFLAGS += -dynamiclib -Wl,-undefined,dynamic_lookup
  LIBBOOT := lib/darwin/libboot.dylib
  LIBHAIL := lib/darwin/libhail_$(SUBMAKE_NAME).dylib
endif

# The libboot will load first and is always an abi_v2 so that it works with
# either old or new libstdc++
ifneq ($(SUBMAKE_NAME),abi_v2)
  LIBBOOT :=
endif

all: $(BUILD) $(LIBBOOT) $(LIBHAIL)

$(BUILD):
	-mkdir -p $@

debug:
	echo "make debug SUBMAKE_NAME=$(SUBMAKE_NAME)"
ifndef JAVA_HOME
	echo JAVA_HOME undefined
endif
	echo "JAVA_HOME is $(JAVA_HOME)"
	echo "CXX is $(CXX)"
	-$(CXX) --version

$(BUILD)/functional-tests: ibs.cpp test.cpp
	mkdir -p build
	$(CXX) $(CXXFLAGS) -DNUMBER_OF_GENOTYPES_PER_ROW=256 ibs.cpp test.cpp -o $(BUILD)/functional-tests

test: $(BUILD)/functional-tests
	./$(BUILD)/functional-tests

clean:
	-rm -rf $(BUILD) headers lib $(LIBBOOT) $(LIBHAIL)
	echo done

$(BUILD)/headers: $(shell /usr/bin/find ../resources/include -name "*.h")
	touch $@

$(BUILD_OBJECTS) $(BUILD)/NativeBoot.o: $(BUILD)/headers

ALL_HEADER_FILES := $(shell /usr/bin/find ../resources/include -name "*.h")
ALL_HEADER_CKSUM := $(shell /bin/cat $(ALL_HEADER_FILES) | /usr/bin/cksum | /usr/bin/cut -d " " -f 1)

$(BUILD)/NativeModule.o: NativeModule.cpp
	$(CXX) $(CXXFLAGS) -DALL_HEADER_CKSUM=$(ALL_HEADER_CKSUM)UL -c NativeModule.cpp -o $@

# if your build machine cannot access this URL, download this tar.gz and place
# it in the same directory as this Makefile. If you do so, the build will
# succeed without attempting to access the Internet.
# WGET ?= wget
WGET ?= curl -L -O
libsimdpp-2.0-rc2.tar.gz:
	${WGET} https://storage.googleapis.com/hail-common/libsimdpp-2.0-rc2.tar.gz

libsimdpp-2.0-rc2: libsimdpp-2.0-rc2.tar.gz
	tar -xzf libsimdpp-2.0-rc2.tar.gz

ifneq ($(LIBBOOT),)
$(LIBBOOT): $(BUILD)/NativeBoot.o
	-mkdir -p $(basename $(LIBBOOT))
	$(CXX) $(LIBFLAGS) $(LIBDIRS) $(CXXFLAGS) $(BUILD)/NativeBoot.o -o $@
	cp -p $@ $(PREBUILT)/$@
endif

$(LIBHAIL): libsimdpp-2.0-rc2 $(BUILD_OBJECTS)
	-mkdir -p $(basename $(LIBHAIL))
	$(CXX) $(LIBFLAGS) $(LIBDIRS) $(CXXFLAGS) $(BUILD_OBJECTS) -o $@ 
	cp -p $@ $(PREBUILT)/$@

endif
