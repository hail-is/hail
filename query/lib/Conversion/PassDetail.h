//===- PassDetail.h - Conversion Pass class details -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CONVERSION_PASSDETAIL_H_
#define CONVERSION_PASSDETAIL_H_

#include "mlir/Pass/Pass.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/FunctionInterfaces.h"

namespace hail {
namespace ir {
class SandboxDialect;
} // namespace ir

#define GEN_PASS_CLASSES
#include "Conversion/Passes.h.inc"

} // namespace hail

#endif // CONVERSION_PASSDETAIL_H_
