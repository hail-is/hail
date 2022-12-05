#ifndef HAIL_TRANSFORMS_PASSES_H
#define HAIL_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"
#include <limits>

namespace hail::ir {

std::unique_ptr<mlir::Pass> createTestMissingnessAnalysisPass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "hail/Transforms/Passes.h.inc"

} // namespace hail::ir

#endif // HAIL_TRANSFORMS_PASSES_H
