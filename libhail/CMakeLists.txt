cmake_minimum_required(VERSION 3.16)

# project name
project(libhail)

# c++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# -fPIC for everything
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(HAIL_PYTHON_MODULE_NAME "_hail.so" CACHE STRING "full name (including .so) of the hail python module library, useful for python packaging")
set(HAIL_PYTHON_MODULE_LIBDIR "" CACHE STRING "directory to put the python module library into on install, useful for python packaging")

find_package(LLVM REQUIRED CONFIG)
find_package(Python COMPONENTS Interpreter Development)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

set(lib_sources
  src/hail/allocators.cpp
  src/hail/format.cpp
  src/hail/type.cpp
  src/hail/value.cpp
  src/hail/vtype.cpp
  # query
  src/hail/query/ir.cpp
  src/hail/query/ir_type.cpp
  src/hail/query/backend/compile.cpp
  src/hail/query/backend/jit.cpp
  src/hail/query/backend/stype.cpp
  src/hail/query/backend/svalue.cpp
  # runtime
  src/hail/runtime/runtime.cpp)
set(python_module_sources
  src/hail/python/_hail/module.cpp)
set(test_sources
  src/hail/test_format.cpp
  src/hail/test_value.cpp
  src/hail/test.cpp)

add_library(hail_objlib OBJECT ${lib_sources})
add_executable(test $<TARGET_OBJECTS:hail_objlib> ${test_sources})
add_library(hail_python SHARED
  $<TARGET_OBJECTS:hail_objlib>
  ${python_module_sources})

set_target_properties(hail_python
  PROPERTIES
    PREFIX ""
    SUFFIX ""
    OUTPUT_NAME ${HAIL_PYTHON_MODULE_NAME})

target_include_directories(hail_objlib PRIVATE "src")
target_include_directories(hail_python PRIVATE ${Python_INCLUDE_DIRS} PRIVATE "src")
target_include_directories(test PRIVATE "src")

llvm_config(test x86asmparser x86codegen orcjit)
llvm_config(hail_python x86asmparser x86codegen orcjit)

target_link_libraries(hail_python PUBLIC ${Python_LIBRARIES})
install(TARGETS hail_python LIBRARY DESTINATION ${HAIL_PYTHON_MODULE_LIBDIR})
