# [MLIR](https://mlir.llvm.org) + [hail](https://hail.is) = 🚀🧬?

## Dependencies

You will need `cmake` and `ninja`. If you're on OS X, use https://brew.sh:

```sh
brew install cmake ninja
```

## Building/Installing LLVM and MLIR
Obviously, update the paths for your environment. `$HAIL_DIR` is the root of the
hail repository.

```sh
git clone https://github.com/llvm/llvm-project.git
mkdir llvm-project/build
cd llvm-project/build
git checkout llvmorg-15.0.3  # latest stable LLVM/MLIR release

# Some notes:
#     1. -G Ninja generates a build.ninja file rather than makefiles it's not
#        required but is recommended by LLVM
#     2. The CMAKE_INSTALL_PREFIX I put here is a subdirectory of the mlir-hail
#        (this repo's) root. If you do this, add that directory to
#        .git/info/exclude and it will be like adding it to a gitignore
#     3. On linux, using lld via -DLLVM_ENABLE_LLD=ON can speed up the build due
#        to faster linking.
#
# The -DLLVM_BUILD_EXAMPLES=ON flag is optional.
cmake ../llvm -G Ninja \
   -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra;mlir" \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="AArch64;X86;NVPTX;AMDGPU" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
   -DCMAKE_INSTALL_PREFIX=$HAIL_DIR/query/.dist/llvm \
   -DCMAKE_EXPORT_COMPILE_COMMANDS=1
ninja # this will take a while
ninja install
```

## Building the hail-query-mlir project

To set up the build, from this directory:

```sh
mkdir build
cd build
# You can set CC and CXX or use -DCMAKE_C_COMPILER and -DCMAKE_CXX_COMPILER to
# change the C and C++ compilers used to build this project.
cmake .. -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DMLIR_DIR=$HAIL_DIR/.dist/llvm/lib/cmake/mlir \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
  -DLLVM_BUILD_BINARY_DIR=~/src/llvm-project/build/bin \
  # ^ this argument is necessary to find llvm-lit and FileCheck for the tests
  #   they are skipped and a warning printed otherwise
```

To build:
```sh
cd build
ninja
```
