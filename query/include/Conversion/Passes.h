#ifndef CONVERSION_PASSES_H
#define CONVERSION_PASSES_H

#include "Conversion/CPSToCF/CPSToCF.h"
#include "Conversion/LowerSandbox/LowerSandbox.h"
#include "Conversion/LowerToLLVM/LowerToLLVM.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"

namespace hail {

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "Conversion/Passes.h.inc"

} // namespace hail

#endif // CONVERSION_PASSES_H