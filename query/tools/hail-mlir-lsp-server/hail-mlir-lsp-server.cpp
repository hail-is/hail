//===- mlir-lsp-server.cpp - MLIR Language Server -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

#include "InitAllDialects.h"

using namespace hail::ir;

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registerAllDialects(registry);

  return failed(MlirLspServerMain(argc, argv, registry));
}
