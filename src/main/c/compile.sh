#!/bin/bash
set -ex

clang -O3 -march=native -g -dynamiclib -std=gnu99 ibs.c -current_version 1.0.1 -compatibility_version 1.0.1 -fvisibility=hidden -o ibs.dylib

cp ibs.dylib ../../../build/resources/main/darwin/libibs.dylib
cp ibs.dylib ../../../build/resources/main/libibs.dylib
