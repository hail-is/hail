//===- PassDetail.h - Transforms Pass class details -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HAIL_TRANSFORMS_PASSDETAIL_H_
#define HAIL_TRANSFORMS_PASSDETAIL_H_

#include "mlir/Pass/Pass.h"

namespace hail::ir {
#define GEN_PASS_CLASSES
#include "hail/Transforms/Passes.h.inc"
} // namespace hail::ir

#endif // HAIL_TRANSFORMS_PASSDETAIL_H_
