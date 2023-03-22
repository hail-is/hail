cmake_minimum_required(VERSION 3.16)
project(hail-mlir-dialect LANGUAGES CXX C)

# CMP0116: Ninja generators transform `DEPFILE`s from `add_custom_command()`
# New in CMake 3.20. https://cmake.org/cmake/help/latest/policy/CMP0116.html
if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ standard to conform to")

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(HAIL_MAIN_SRC_DIR     ${CMAKE_CURRENT_SOURCE_DIR}  )
set(HAIL_MAIN_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include )

list(INSERT CMAKE_MODULE_PATH 0
  "${HAIL_MAIN_SRC_DIR}/cmake/modules"
)

include(AddHail)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

if(MLIR_ENABLE_BINDINGS_PYTHON)
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
endif()

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# supress a warning on clang
add_compile_options($<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:-Wno-ambiguous-reversed-operator>)

if(HAIL_USE_CLANG_TIDY AND NOT CMAKE_CXX_CLANG_TIDY)
    find_program(CLANG_TIDY "clang-tidy" HINTS ${LLVM_TOOLS_BINARY_DIR})
    if(CLANG_TIDY)
        message(STATUS "Found clang-tidy: ${CLANG_TIDY}")
        set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY} --use-color --warnings-as-errors=*)
    endif()
endif()

add_subdirectory(include/hail)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(test)
