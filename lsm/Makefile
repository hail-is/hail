MAKEFLAGS += --no-builtin-rules
.SUFFIXES:
.DEFAULT_GOAL := build/main

CXXFLAGS += -std=c++2a -fPIC -ggdb -g -fno-strict-aliasing
CXXFLAGS += -fno-omit-frame-pointer  # enables libasan to provide complete stack traces
CXXFLAGS += -Wall -Werror -Wextra -Wpedantic -Wno-unused-parameter
CXXFLAGS += -Og -fsanitize=address -fsanitize=undefined

CPP_FILES := $(shell find . -iname '*.cpp' -print | sed 's:./::')
OBJECTS := $(CPP_FILES:%.cpp=build/%.o)

.SECONDARY: $(OBJECTS)  # do not delete the object files (useful for debugging)

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

.PHONY: test
test: build/main
	./test.sh
