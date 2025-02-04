#!/bin/sh
# KING 2.2.5, compiled with Homebrew GCC 10.2.0 on macOS 10.14.6
set -ex
king --kinship -b fastlmmTest.bed --prefix fastlmmTest
bgzip fastlmmTest.kin0
mv fastlmmTest.kin0.gz fastlmmTest.kin0.bgz
