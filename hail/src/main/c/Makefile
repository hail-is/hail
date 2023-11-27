MAKEFLAGS += --no-builtin-rules
.PHONY: all clean debug prebuilt test reset-prebuilt
.SUFFIXES:
.DEFAULT_GOAL := all

UNAME_S :=$(shell uname -s)
UNAME_P :=$(shell uname -p)
UNAME_M :=$(shell uname -m)

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

# before libsimdpp and catch.hpp are downloaded, clang -MG -MM will generate
# unresolved dependencies
.PHONY: simdpp/simd.h catch.hpp
simdpp/simd.h: $(LIBSIMDPP)
catch.hpp: $(CATCH_HEADER_LOCATION)


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
else ifeq ($(UNAME_S),Darwin)
  BREW_PREFIX := $(shell brew --prefix)
  ifneq ($(BREW_PREFIX),)
    CXXFLAGS += -I$(BREW_PREFIX)/include
  endif

  LIBFLAGS += -dynamiclib -Wl,-undefined,dynamic_lookup
  ifeq ($(UNAME_M),arm64)
    LIBBOOT_X86_64 := lib/darwin/x86_64/libboot.dylib
    LIBHAIL_X86_64 := lib/darwin/x86_64/libhail.dylib
    LIBBOOT_ARM64 := lib/darwin/arm64/libboot.dylib
    LIBHAIL_ARM64 := lib/darwin/arm64/libhail.dylib
    LIBBOOT := lib/darwin/libboot.dylib
    LIBHAIL := lib/darwin/libhail.dylib
  else
    LIBBOOT := lib/darwin/libboot.dylib
    LIBHAIL := lib/darwin/libhail.dylib
  endif
endif

BUILD_X86_64 := $(BUILD)/x86_64
CXXFLAGS_X86_64 := $(CXXFLAGS)
LIBFLAGS_X86_64 := $(LIBFLAGS)

BUILD_ARM64 := $(BUILD)/arm64
CXXFLAGS_ARM64 := $(CXXFLAGS)
LIBFLAGS_ARM64 := $(LIBFLAGS)

ifeq ($(UNAME_M),arm64)
  NATIVE_BUILD := $(BUILD_ARM64)
else
  NATIVE_BUILD := $(BUILD_X86_64)
endif

# If you want to add a new cpp file, like foo.cpp, to the library, add foo to
# this list. Remember to rerun make prebuilt.
OBJECTS := \
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

BUILD_OBJECTS_X86_64 := $(addprefix $(BUILD_X86_64)/, $(OBJECTS:%=%.o))
BUILD_OBJECTS_ARM64 := $(addprefix $(BUILD_ARM64)/, $(OBJECTS:%=%.o))

TEST_CPP := $(wildcard *_test.cpp) testutils/unit-tests.cpp
TEST_OBJECTS_X86_64 := $(foreach file,$(TEST_CPP),$(BUILD_X86_64)/$(basename $(file)).o)
TEST_OBJECTS_ARM64 := $(foreach file,$(TEST_CPP),$(BUILD_ARM64)/$(basename $(file)).o)

ALL_CPP := $(shell find * -iname '*.cpp')

ifeq ($(UNAME_S)-$(UNAME_M),Darwin-arm64)
-include $(addprefix $(BUILD_X86_64)/, $(ALL_CPP:%.cpp=%.d))
-include $(addprefix $(BUILD_ARM64)/, $(ALL_CPP:%.cpp=%.d))
else ifeq ($(UNAME_M),arm64)
-include $(addprefix $(BUILD_ARM64)/, $(ALL_CPP:%.cpp=%.d))
else ifeq ($(UNAME_M),x86_64)
-include $(addprefix $(BUILD_X86_64)/, $(ALL_CPP:%.cpp=%.d))
endif

# If no inherited "-march", and we're not building a universal dylib on Apple Silicon, then use
# "-march=sandybridge" or "-march=corei7-avx" for ISA compatibility with MacBook Pro's since 2011
# (also the earliest cpu with AVX). Fall back to "-march=native" if the compiler doesn't support any
# of those.
#
# Otherwise, we're on Apple Silicon and building a universal dylib. We know exactly which -march
# settings are valid.
ifeq ($(filter -march=%,$(CXXFLAGS)),)
  ifeq ($(UNAME_S)-$(UNAME_M),Darwin-arm64)
    CXXFLAGS_X86_64 += -arch x86_64 -march=sandybridge
    LIBFLAGS_X86_64 += -arch x86_64
    CXXFLAGS_ARM64 += -arch arm64 -mcpu=apple-m1
    LIBFLAGS_ARM64 += -arch arm64
  else
    FAIL_A_X86_64 :=$(shell cp /dev/null a.cpp; $(CXX) -march=sandybridge -c a.cpp 2>&1 || echo FAIL; rm -f a.cpp a.o)
    ifeq ($(FAIL_A_X86_64),)
      CXXFLAGS_X86_64 += -march=sandybridge
    else
      # g++-4.8.x accepts "-march=corei7-avx" but not "-march=sandybridge	"
      FAIL_B_X86_64 :=$(shell cp /dev/null a.cpp; $(CXX) -march=corei7-avx -c a.cpp 2>&1 || echo FAIL; rm -f a.cpp a.o)
      ifeq ($(FAIL_B_X86_64),)
        CXXFLAGS_X86_64 += -march=corei7-avx
      else
        CXXFLAGS_X86_64 += -march=native
      endif
    endif

    CXXFLAGS_ARM64 += -march=armv8.5-a  # we do not test on arm chips
  endif
endif

$(BUILD_X86_64)/%.d: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_X86_64) $< -MG -M -MF $@ -MT $(@:%.d=%.o)

$(BUILD_ARM64)/%.d: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_ARM64) $< -MG -M -MF $@ -MT $(@:%.d=%.o)

$(BUILD_X86_64)/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) -o $@ $(CXXFLAGS_X86_64) -MD -MF $(@:%.o=%.d) -MT $@ -c $<

$(BUILD_ARM64)/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) -o $@ $(CXXFLAGS_ARM64) -MD -MF $(@:%.o=%.d) -MT $@ -c $<

all: $(LIBBOOT) $(LIBHAIL)

debug:
	echo "make debug"
ifndef JAVA_HOME
	echo JAVA_HOME undefined
endif
	echo "JAVA_HOME is $(JAVA_HOME)"
	echo "CXX is $(CXX)"
	-$(CXX) --version

$(BUILD_X86_64)/functional-tests: ibs.cpp test.cpp $(LIBSIMDPP)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -DNUMBER_OF_GENOTYPES_PER_ROW=256 -o $(BUILD_X86_64)/functional-tests ibs.cpp test.cpp

$(BUILD_ARM64)/functional-tests: ibs.cpp test.cpp $(LIBSIMDPP)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -DNUMBER_OF_GENOTYPES_PER_ROW=256 -o $(BUILD_ARM64)/functional-tests ibs.cpp test.cpp

$(BUILD_X86_64)/unit-tests: $(BUILD_OBJECTS_X86_64) $(TEST_OBJECTS_X86_64)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -o $(BUILD_X86_64)/unit-tests $(BUILD_OBJECTS_X86_64) $(TEST_OBJECTS_X86_64) -ldl

$(BUILD_ARM64)/unit-tests: $(BUILD_OBJECTS_ARM64) $(TEST_OBJECTS_ARM64)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -o $(BUILD_ARM64)/unit-tests $(BUILD_OBJECTS_ARM64) $(TEST_OBJECTS_ARM64) -ldl

prebuilt: $(LIBBOOT) $(LIBHAIL)
	@mkdir -p $(PREBUILT)/$(dir $<)
	cp -p -f $^ $(PREBUILT)/$(dir $<)

reset-prebuilt:
	git checkout HEAD -- $(PREBUILT)/$(LIBBOOT)
	git checkout HEAD -- $(PREBUILT)/$(LIBHAIL)

test: $(NATIVE_BUILD)/functional-tests $(NATIVE_BUILD)/unit-tests
	./$(NATIVE_BUILD)/unit-tests -w NoAssertions -s -d yes -# --use-colour yes -r xml -o $(NATIVE_BUILD)/cxx-test.xml; \
			case "$$?" in \
				*) \
				mkdir -p $(NATIVE_BUILD)/reports; \
				cp testutils/style.css $(NATIVE_BUILD)/reports; \
				xsltproc -o $(NATIVE_BUILD)/reports/index.html testutils/test-reporter.xslt $(NATIVE_BUILD)/cxx-test.xml;; \
			esac
	./$(NATIVE_BUILD)/functional-tests

benchmark: $(NATIVE_BUILD)/unit-tests
	./$(NATIVE_BUILD)/unit-tests "[!benchmark]" -s -d yes -# -r xml -o $(NATIVE_BUILD)/cxx-benchmark.xml; \
			case "$$?" in \
				*) \
				mkdir -p $(NATIVE_BUILD)/benchmark-reports; \
				cp testutils/style.css $(NATIVE_BUILD)/benchmark-reports; \
				xsltproc -o $(NATIVE_BUILD)/benchmark-reports/index.html testutils/test-reporter.xslt $(NATIVE_BUILD)/cxx-benchmark.xml;; \
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

$(BUILD_X86_64)/NativeModule.o: NativeModule.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_X86_64) -DALL_HEADER_CKSUM=$(ALL_HEADER_CKSUM)UL -c NativeModule.cpp -o $@

$(BUILD_ARM64)/NativeModule.o: NativeModule.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_ARM64) -DALL_HEADER_CKSUM=$(ALL_HEADER_CKSUM)UL -c NativeModule.cpp -o $@

$(CATCH_HEADER_LOCATION):
	@mkdir -p $(@D)
	curl -sSL 'https://github.com/catchorg/Catch2/releases/download/v2.6.0/catch.hpp' > $@

$(LIBSIMDPP).tar.gz:
	curl -sSL https://storage.googleapis.com/hail-common/$@ > $@

$(LIBSIMDPP): $(LIBSIMDPP).tar.gz
	tar -xzf $<

ifeq ($(UNAME_S)-$(UNAME_M),Darwin-arm64)
$(LIBBOOT_X86_64): $(BUILD_X86_64)/NativeBoot.o
	@mkdir -p $(dir $@)
	$(CXX) $(LIBFLAGS_X86_64) $(LIBDIRS) $(CXXFLAGS) $^ -o $@
$(LIBHAIL_X86_64): $(BUILD_OBJECTS_X86_64)
	@mkdir -p $(dir $@)
	$(CXX) $(LIBFLAGS_X86_64) $(LIBDIRS) $(CXXFLAGS) $^ -o $@

$(LIBBOOT_ARM64): $(BUILD_ARM64)/NativeBoot.o
	@mkdir -p $(dir $@)
	$(CXX) $(LIBFLAGS_ARM64) $(LIBDIRS) $(CXXFLAGS) $^ -o $@
$(LIBHAIL_ARM64): $(BUILD_OBJECTS_ARM64)
	@mkdir -p $(dir $@)
	$(CXX) $(LIBFLAGS_ARM64) $(LIBDIRS) $(CXXFLAGS) $^ -o $@

$(LIBBOOT): $(LIBBOOT_X86_64) $(LIBBOOT_ARM64)
	@mkdir -p $(dir $@)
	lipo -create $^ -output $@
$(LIBHAIL): $(LIBHAIL_X86_64) $(LIBHAIL_ARM64)
	@mkdir -p $(dir $@)
	lipo -create $^ -output $@
else
$(LIBBOOT): $(BUILD_X86_64)/NativeBoot.o
	@mkdir -p $(dir $@)
	$(CXX) $(LIBFLAGS_X86_64) $(LIBDIRS) $(CXXFLAGS) $^ -o $@
$(LIBHAIL): $(BUILD_OBJECTS_X86_64)
	@mkdir -p $(dir $@)
	$(CXX) $(LIBFLAGS_X86_64) $(LIBDIRS) $(CXXFLAGS) $^ -o $@
endif
