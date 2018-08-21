# This makefile invokes itself recursively to build the libraries,
# possibly with different compiler options.
#
# If compiling with g++, we can specify the abi-version to keep
# maximum compatibility when linked against old versions of libstdc++
#   -fabi-version=2 was the default from g++-3.4 to g++-4.9
#   -fabi-version=9 was the default in g++-5.x

# On MacOS, libc++ is the default library, and the default clang-based
# compiler doesn't accept the -fabi-version=N option.  We still use the 
# name "libhail.dylib" to mean "libhail compatible with recent abi

ifndef SUBMAKE_NAME

.PHONY: top all clean debug prebuilt test

top: all

CXX_VERSION :=$(shell $(CXX) --version | head -1)
ifneq ($(filter clang llvm LLVM,$(CXX_VERSION)),)
  CXX_TYPE :=clang
else
  CXX_TYPE :=not_clang
endif

# If this compiler supports "-fabi-version=9", we will build libhail.so that way.
# If not, you get whatever you get.

CXX_ABI_V9_ERRS :=$(shell echo "" >a.cpp; $(CXX) -fabi-version=9 -c a.cpp 2>a.err; rm -f a.cpp a.o 2>/dev/null; cat a.err)

all clean debug prebuilt test:
ifeq ($(CXX_ABI_V9_ERRS),)
	@$(MAKE) --no-print-directory SUBMAKE_NAME= SUBMAKE_CXXFLAGS=-fabi-version=9 $@
else
	@$(MAKE) --no-print-directory SUBMAKE_NAME= SUBMAKE_CXXFLAGS= $@
endif

else

UNAME_S :=$(shell uname -s)
UNAME_P :=$(shell uname -p)

# Resolve the full pathnames of executables so that any unintentional
# use of non-system executables will be visible in the build log.

CAT :=$(shell which cat 2>/dev/null)
ifeq ($(CAT),)
  CAT :=cat
endif

CKSUM :=$(shell which cksum 2>/dev/null)
ifeq ($(CKSUM),)
  CKSUM :=cksum
endif

CP :=$(shell which cp 2>/dev/null)
ifeq ($(CP),)
  CP :=cp
endif

CURL :=$(shell which curl 2>/dev/null)
ifeq ($(CURL),)
  CURL :=curl
endif

CUT :=$(shell which cut 2>/dev/null)
ifeq ($(CUT),)
  CUT :=cut
endif

FIND :=$(shell which find 2>/dev/null)
ifeq ($(FIND),)
  FIND :=find
endif

MKDIR :=$(shell which mkdir 2>/dev/null)
ifeq ($(MKDIR),)
  MKDIR :=mkdir
endif

RM :=$(shell which rm 2>/dev/null)
ifeq ($(RM),)
  RM :=rm
endif

TAR :=$(shell which tar 2>/dev/null)
ifeq ($(TAR),)
  TAR :=tar
endif

TOUCH :=$(shell which touch 2>/dev/null)
ifeq ($(TOUCH),)
  TOUCH :=touch
endif

.PHONY: all clean debug test libsimdpp

BUILD := build$(SUBMAKE_NAME)
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

BUILD_OBJECTS := $(OBJECTS:%=$(BUILD)/%)

$(BUILD)/%.o: %.cpp
	$(CXX) -o $@ $(CXXFLAGS) $(SUBMAKE_CXXFLAGS) -c $<

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

# Currently source code for libboot and libhail only uses features up to C++11.
# The intention is to support C++17 for dynamic-compiled code eventually, but 
# to maximize compatibility with different compilers/platforms we don't
# require that here.
#
# The code relies heavily on C++11's std::shared_ptr, so you need a compiler
# that supports at least the C++11 standard.

CXXSTD := -std=c++11

# Check for any inherited CXXFLAGS which could interfere with
# ABI compatibility.  Such flags will cause a warning, then will be
# ignored.  This list may not be exhaustive: any options affecting the
# procedure-calling standard or data layout may cause trouble.

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

all: libsimdpp-2.0-rc2 $(BUILD) $(LIBBOOT) $(LIBHAIL)

$(BUILD):
	-$(MKDIR) -p $@

debug:
	echo "make debug SUBMAKE_NAME=$(SUBMAKE_NAME)"
ifndef JAVA_HOME
	echo JAVA_HOME undefined
endif
	echo "JAVA_HOME is $(JAVA_HOME)"
	echo "CXX is $(CXX)"
	-$(CXX) --version

$(BUILD)/functional-tests: ibs.cpp test.cpp
	-$(MKDIR) -p build
	$(CXX) $(CXXFLAGS) -DNUMBER_OF_GENOTYPES_PER_ROW=256 ibs.cpp test.cpp -o $(BUILD)/functional-tests

prebuilt: all
	@echo updating $(PREBUILT) ...
ifneq ($(LIBBOOT),)
	$(CP) -p -f $(LIBBOOT) $(PREBUILT)/$(LIBBOOT)
endif
	$(CP) -p -f $(LIBHAIL) $(PREBUILT)/$(LIBHAIL)

test: $(BUILD)/functional-tests
	./$(BUILD)/functional-tests

clean:
	-rm -rf $(BUILD) lib libsimdpp-2.0-rc2 $(LIBBOOT) $(LIBHAIL)

# The $(BUILD)/headers target is updated whenever any .h file has changed
# under either the source directory, *or* the ../resources/include tree.
# Header files under ../resources/include are visible to both libhail.so
# code and dynamic-generated code; headers under the source are only visible
# to code that goes into libhail.so
#
# We use this as a pessimistic dependency: if any header has changed, all
# objects will be rebuilt. Since the quantity of code is small and the 
# clean-rebuild time is short (compared to other parts of the build process),
# this seems good enough, and simpler than managing precise dependencies.

$(BUILD)/headers: $(shell $(FIND) . -name "*.h"; $(FIND) ../resources/include -name "*.h")
	$(TOUCH) $@

$(BUILD_OBJECTS) $(BUILD)/NativeBoot.o: $(BUILD)/headers

# We take all headers files visible to dynamic-generated code, together with
# the output of "$(CXX) --version", to give a checksum $(ALL_HEADER_CKSUM)
# which is then used to modify NativeModule's hash function.  This gives very
# high probability that any changes to either the C++ compiler, or the header
# files, will cause recompilation of dynamic-generated C++ rather than getting
# an erroneous cache hit on outdated DLL files.

ALL_HEADER_FILES := $(shell $(FIND) ../resources/include -name "*.h")
ALL_HEADER_CKSUM := $(shell $(CXX) --version >.cxx.vsn ; $(CAT) .cxx.vsn $(ALL_HEADER_FILES) | $(CKSUM) | $(CUT) -d " " -f 1)

$(BUILD)/NativeModule.o: NativeModule.cpp
	$(CXX) $(CXXFLAGS) -DALL_HEADER_CKSUM=$(ALL_HEADER_CKSUM)UL -c NativeModule.cpp -o $@

# If your build machine cannot access this URL, download this tar.gz and place
# it in the same directory as this Makefile. If you do so, the build will
# succeed without attempting to access the Internet.
# WGET ?= wget
WGET ?= $(CURL) -L -O
libsimdpp-2.0-rc2.tar.gz:
	$(WGET) https://storage.googleapis.com/hail-common/libsimdpp-2.0-rc2.tar.gz

libsimdpp-2.0-rc2: libsimdpp-2.0-rc2.tar.gz
	$(TAR) -xzf libsimdpp-2.0-rc2.tar.gz

ifneq ($(LIBBOOT),)
$(LIBBOOT): $(BUILD)/NativeBoot.o
	-$(MKDIR) -p $(basename $(LIBBOOT))
	$(CXX) $(LIBFLAGS) $(LIBDIRS) $(CXXFLAGS) $(BUILD)/NativeBoot.o -o $@
endif

$(LIBHAIL): $(BUILD_OBJECTS)
	-$(MKDIR) -p $(basename $(LIBHAIL))
	$(CXX) $(LIBFLAGS) $(LIBDIRS) $(CXXFLAGS) $(BUILD_OBJECTS) -o $@ 

endif
