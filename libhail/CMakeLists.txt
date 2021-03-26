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

add_executable(test
  src/hail/allocators.cpp
  src/hail/format.cpp
  src/hail/test_format.cpp
  src/hail/type.cpp
  src/hail/value.cpp
  src/hail/test_value.cpp
  src/hail/vtype.cpp
  # query
  src/hail/query/ir.cpp
  src/hail/query/ir_type.cpp
  src/hail/query/backend/compile.cpp
  src/hail/query/backend/jit.cpp
  src/hail/query/backend/stype.cpp
  src/hail/query/backend/svalue.cpp
  # runtime
  src/hail/runtime/runtime.cpp
  src/hail/test.cpp)

llvm_config(test x86asmparser x86codegen orcjit)

target_include_directories(test PRIVATE "src")

target_link_libraries(test ${llvm_libraries})
