# [MLIR](https://mlir.llvm.org) + [hail](https://hail.is) = 🚀🧬?

## Building/Installing LLVM and MLIR
Obviously, update the paths for your environment.

```sh
git clone https://github.com/llvm/llvm-project.git
mkdir llvm-project/build
cd llvm-project/build
git checkout llvmorg-14.0.6  # latest stable LLVM/MLIR release

# Some notes:
#     1. -G Ninja generates a build.ninja file rather than makefiles it's not
#        required but is recommended by LLVM
#     2. The CMAKE_INSTALL_PREFIX I put here is a subdirectory of the mlir-hail
#        (this repo's) root. If you do this, add that directory to
#        .git/info/exclude and it will be like adding it to a gitignore
#     3. On linux, using lld via -DLLVM_ENABLE_LLD=ON can speed up the build due
#        to faster linking.
cmake ../llvm -G Ninja \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \ # this is optional
   -DLLVM_TARGETS_TO_BUILD="AArch64;X86;NVPTX;AMDGPU" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
   -DCMAKE_INSTALL_PREFIX=~/src/hail/query-mlir/.dist/llvm
ninja # this will take a while
ninja install
```

## Building the hail-query-mlir project

To set up the build:

```sh
mkdir build
cd build
# same prefix as the install directory above in the build/install LLVM
cmake .. -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DMLIR_DIR=../.dist/llvm/lib/cmake/mlir \
  -DLLVM_BUILD_BINARY_DIR=~/src/llvm-project/build/bin
  # ^ this argument is necessary to find llvm-lit and FileCheck for the tests
  #   they are skipped and a warning printed otherwise
```

To build:
```sh
cd build
ninja
```
