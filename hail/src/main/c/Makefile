MAKEFLAGS += --no-builtin-rules
.PHONY: all clean debug prebuilt test reset-prebuilt
.SUFFIXES:
.DEFAULT_GOAL := all

UNAME_S :=$(shell uname -s)
UNAME_P :=$(shell uname -p)

BUILD := build

# Control optimization level and gdb-in-xterm code in NativePtr.cpp
HAIL_ENABLE_DEBUG := 0

ifeq ($(HAIL_ENABLE_DEBUG),1)
  HAIL_OPT_FLAGS := -O1 -DHAIL_ENABLE_DEBUG=1
else
  HAIL_OPT_FLAGS := -O3
endif

# Change this setting to update the version of libsimdpp
LIBSIMDPP := libsimdpp-2.1

CATCH_HEADER_LOCATION := ../resources/include/catch.hpp

# If you want to add a new cpp file, like foo.cpp, to the library, add foo to
# this list. Remember to rerun make prebuilt.
OBJECTS := \
  davies \
  ibs \
  Decoder \
  Encoder \
  Logging \
  NativeCodeSuite \
  NativeLongFunc \
  NativeModule \
  NativePtr \
  NativeStatus \
  ObjectArray \
  PartitionIterators \
  Region \
  Upcalls \
  FS \

BUILD_OBJECTS := $(OBJECTS:%=$(BUILD)/%.o)

# before libsimdpp and catch.hpp are downloaded, clang -MG -MM will generate
# unresolved dependencies
.PHONY: simdpp/simd.h catch.hpp
simdpp/simd.h: $(LIBSIMDPP)
catch.hpp: $(CATCH_HEADER_LOCATION)

TEST_CPP := $(wildcard *_test.cpp) testutils/unit-tests.cpp
TEST_OBJECTS := $(foreach file,$(TEST_CPP),$(BUILD)/$(basename $(file)).o)

ALL_CPP := $(shell find * -iname '*.cpp')
HEADER_DEPENDENCIES := $(ALL_CPP:%.cpp=build/%.d)
-include $(HEADER_DEPENDENCIES)

$(BUILD)/%.d: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $< -MG -M -MF $@ -MT $(@:%.d=%.o)

$(BUILD)/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) -o $@ $(CXXFLAGS) -MD -MF $(@:%.o=%.d) -MT $@ -c $<

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

CXXSTD := -std=c++14

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

# If no inherited "-march=%", then use "-march=sandybridge" or "-march=corei7-avx"
# for ISA compatibility with MacBook Pro's since 2011 (also the earliest cpu with AVX).
# Fall back to "-march=native" if the compiler doesn't support either of those.

ifeq ($(filter -march=%,$(CXXFLAGS)),)
  FAIL_A :=$(shell cp /dev/null a.cpp; $(CXX) -march=sandybridge -c a.cpp 2>&1 || echo FAIL; rm -f a.cpp a.o)
  ifeq ($(FAIL_A),)
    CXXFLAGS += -march=sandybridge
  else
    # g++-4.8.x accepts "-march=corei7-avx" but not "-march=sandybridge	"
    FAIL_B :=$(shell cp /dev/null a.cpp; $(CXX) -march=corei7-avx -c a.cpp 2>&1 || echo FAIL; rm -f a.cpp a.o)
    ifeq ($(FAIL_B),)
      CXXFLAGS += -march=corei7-avx
    else
      CXXFLAGS += -march=native
    endif
  endif
endif

# Append to any inherited flags which survived filtering
CXXFLAGS += $(HAIL_OPT_FLAGS) $(CXXSTD) -I$(LIBSIMDPP) -Wall -Wextra
CXXFLAGS += -fPIC -ggdb -fno-strict-aliasing
CXXFLAGS += -I../resources/include -I$(JAVA_HOME)/include -I$(JAVA_HOME)/include/$(JAVA_MD)
LIBFLAGS += -fvisibility=default
PREBUILT := ../../../prebuilt

ifeq ($(UNAME_S),Linux)
  LIBFLAGS += -rdynamic -shared
  LIBBOOT := lib/linux-x86-64/libboot.so
  LIBHAIL := lib/linux-x86-64/libhail.so
  ifneq ($(filter %86,$(UNAME_P)),)
    LIBBOOT := lib/linux-x86/libboot.so
    LIBHAIL := lib/linux-x86/libhail.so
  endif
endif
ifeq ($(UNAME_S),Darwin)
  LIBFLAGS += -dynamiclib -Wl,-undefined,dynamic_lookup
  LIBBOOT := lib/darwin/libboot.dylib
  LIBHAIL := lib/darwin/libhail.dylib
endif

all: $(LIBBOOT) $(LIBHAIL)

debug:
	echo "make debug"
ifndef JAVA_HOME
	echo JAVA_HOME undefined
endif
	echo "JAVA_HOME is $(JAVA_HOME)"
	echo "CXX is $(CXX)"
	-$(CXX) --version

$(BUILD)/functional-tests: ibs.cpp test.cpp $(LIBSIMDPP)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -DNUMBER_OF_GENOTYPES_PER_ROW=256 -o $(BUILD)/functional-tests ibs.cpp test.cpp

$(BUILD)/unit-tests: $(BUILD_OBJECTS) $(TEST_OBJECTS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -o $(BUILD)/unit-tests $(BUILD_OBJECTS) $(TEST_OBJECTS) -ldl

prebuilt: $(LIBBOOT) $(LIBHAIL)
	cp -p -f $^ $(PREBUILT)/$(dir $<)

reset-prebuilt:
	git checkout HEAD -- $(PREBUILT)/$(LIBBOOT)
	git checkout HEAD -- $(PREBUILT)/$(LIBHAIL)

test: $(BUILD)/functional-tests $(BUILD)/unit-tests
	./$(BUILD)/unit-tests -w NoAssertions -s -d yes -# --use-colour yes -r xml -o $(BUILD)/cxx-test.xml; \
			case "$$?" in \
				*) \
				mkdir -p $(BUILD)/reports; \
				cp testutils/style.css $(BUILD)/reports; \
				xsltproc -o $(BUILD)/reports/index.html testutils/test-reporter.xslt $(BUILD)/cxx-test.xml;; \
			esac
	./$(BUILD)/functional-tests

benchmark: $(BUILD)/unit-tests
	./$(BUILD)/unit-tests "[!benchmark]" -s -d yes -# -r xml -o $(BUILD)/cxx-benchmark.xml; \
			case "$$?" in \
				*) \
				mkdir -p $(BUILD)/benchmark-reports; \
				cp testutils/style.css $(BUILD)/benchmark-reports; \
				xsltproc -o $(BUILD)/benchmark-reports/index.html testutils/test-reporter.xslt $(BUILD)/cxx-benchmark.xml;; \
			esac

clean:
	-rm -rf $(BUILD) $(LIBSIMDPP) lib

# We take all headers files visible to dynamic-generated code, together with
# the output of "$(CXX) --version", to give a checksum $(ALL_HEADER_CKSUM)
# which is then used to modify NativeModule's hash function.  This gives very
# high probability that any changes to either the C++ compiler, or the header
# files, will cause recompilation of dynamic-generated C++ rather than getting
# an erroneous cache hit on outdated DLL files.

ALL_HEADER_FILES := $(shell find ../resources/include -name "*.h")
ALL_HEADER_CKSUM := $(shell $(CXX) --version >.cxx.vsn ; cat .cxx.vsn $(ALL_HEADER_FILES) | cksum | cut -d " " -f 1)

$(BUILD)/NativeModule.o: NativeModule.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -DALL_HEADER_CKSUM=$(ALL_HEADER_CKSUM)UL -c NativeModule.cpp -o $@

$(CATCH_HEADER_LOCATION):
	@mkdir -p $(@D)
	curl -sSL 'https://github.com/catchorg/Catch2/releases/download/v2.6.0/catch.hpp' > $@

$(LIBSIMDPP).tar.gz:
	curl -sSL https://storage.googleapis.com/hail-common/$@ > $@

$(LIBSIMDPP): $(LIBSIMDPP).tar.gz
	tar -xzf $<

$(LIBBOOT): $(BUILD)/NativeBoot.o
	@mkdir -p $(dir $(LIBBOOT))
	$(CXX) $(LIBFLAGS) $(LIBDIRS) $(CXXFLAGS) $(BUILD)/NativeBoot.o -o $@

$(LIBHAIL): $(BUILD_OBJECTS)
	@mkdir -p $(dir $(LIBHAIL))
	$(CXX) $(LIBFLAGS) $(LIBDIRS) $(CXXFLAGS) $(BUILD_OBJECTS) -o $@
