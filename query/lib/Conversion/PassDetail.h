// NOLINTNEXTLINE(llvm-header-guard)
#ifndef HAIL_CONVERSION_PASSDETAIL_H_
#define HAIL_CONVERSION_PASSDETAIL_H_

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace hail {
namespace ir {
class SandboxDialect;
} // namespace ir

#define GEN_PASS_CLASSES
#include "hail/Conversion/Passes.h.inc"

} // namespace hail

#endif // HAIL_CONVERSION_PASSDETAIL_H_
