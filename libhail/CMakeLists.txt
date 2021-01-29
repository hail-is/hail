cmake_minimum_required(VERSION 3.16)

# c++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# project name
project(libhail)

find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

add_executable(main src/hail/main.cpp src/hail/format.cpp src/hail/type.cpp src/hail/allocators.cpp src/hail/value.cpp src/hail/ptype.cpp src/hail/ir.cpp src/hail/jit.cpp)

llvm_config(main x86asmparser x86codegen orcjit)

target_include_directories(main PRIVATE "src")

target_link_libraries(main ${llvm_libraries})
