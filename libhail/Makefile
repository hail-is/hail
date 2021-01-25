CPP_FILES = src/hail/main.cpp src/hail/format.cpp src/hail/type.cpp
HPP_FILES = src/hail/allocators.hpp src/hail/format.hpp src/hail/tunion.hpp src/hail/type.hpp

main: $(CPP_FILES) $(HPP_FILES)
	clang++ -g -Wall -std=c++20 -Isrc $(CPP_FILES) -o main

clean:
	rm -f main
