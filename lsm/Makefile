MAKEFLAGS += --no-builtin-rules
.SUFFIXES:
.DEFAULT_GOAL := build/main

CC ?= /usr/local/Cellar/gcc/10.2.0/bin/gcc-10
CXX ?= /usr/local/Cellar/gcc/10.2.0/bin/g++-10
LD ?= /usr/local/Cellar/gcc/10.2.0/bin/gcc-10
LDXX ?= /usr/local/Cellar/gcc/10.2.0/bin/g++-10

CXXFLAGS += -std=c++2a -fPIC -ggdb -g -fno-strict-aliasing
CXXFLAGS += -fno-omit-frame-pointer  # enables libasan to provide complete stack traces
CXXFLAGS += -Wall -Werror -Wextra -Wpedantic -Wno-unused-parameter
CXXFLAGS += -Og -fsanitize=address -fsanitize=undefined

CPP_FILES := $(shell find . -iname '*.cpp' -print)
OBJECTS := $(CPP_FILES:%.cpp=build/%.o)

ifneq ($(MAKECMDGOALS),clean)
-include $(OBJECTS:%.o=%.d)
-include $(PROFILE_OBJECTS:%.o=%.d)
-include $(SMALL_OBJECTS:%.o=%.d)
endif

.PHONY: ctags
ctags:
	rm -rf TAGS
	ctags -e -R .

build/%.d: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $< -MG -M -MF $@ -MT $(@:%.d=%.o)

build/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -MD -MF $(@:%.o=%.d) -MT $@ -c $< -o $@

build/%: build/%.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) \
		-ldl \
		-o $@ \
		$(filter %.o,$^)

.PHONY: clean
clean:
	rm -rf build
