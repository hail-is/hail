cmake_minimum_required(VERSION 3.16)

# project name
project(libhail)

# c++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# -fPIC for everything
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(HAIL_PYTHON_MODULE_LIBDIR "." CACHE STRING "directory to put the python module library into on install, useful for python packaging")

find_package(LLVM CONFIG)
find_package(Python COMPONENTS Interpreter Development)

if (${LLVM_FOUND})
  message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
  include_directories(${LLVM_INCLUDE_DIRS})
  add_definitions(${LLVM_DEFINITIONS})
else()
  message(STATUS "LLVM Not Found, skipping LLVM components")
endif()

include_directories("src")

set(lib_sources
  src/hail/allocators.cpp
  src/hail/format.cpp
  src/hail/type.cpp
  src/hail/value.cpp
  src/hail/vtype.cpp
  # runtime
  src/hail/runtime/runtime.cpp)
set(llvm_backend_sources
  src/hail/query/ir.cpp
  src/hail/query/ir_type.cpp
  src/hail/query/backend/compile.cpp
  src/hail/query/backend/jit.cpp
  src/hail/query/backend/stype.cpp
  src/hail/query/backend/svalue.cpp)
set(python_module_sources
  src/hail/python/_hail/module.cpp)
set(test_sources
  src/hail/test_format.cpp
  src/hail/test_value.cpp
  src/hail/test.cpp)

add_library(hail_objlib OBJECT ${lib_sources})
if (${LLVM_FOUND})
  add_library(llvm_backend_objlib OBJECT ${llvm_backend_sources})
  add_executable(test
    $<TARGET_OBJECTS:hail_objlib>
    $<TARGET_OBJECTS:llvm_backend_objlib>
    ${test_sources})
  llvm_config(test x86asmparser x86codegen orcjit)
endif()

Python_add_library(_hail MODULE WITH_SOABI
  $<TARGET_OBJECTS:hail_objlib>
  ${python_module_sources})

install(TARGETS _hail LIBRARY DESTINATION ${HAIL_PYTHON_MODULE_LIBDIR})
