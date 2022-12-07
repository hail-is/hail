#ifndef HAIL_CONVERSION_CPSTOCF_CPSTOCF_H
#define HAIL_CONVERSION_CPSTOCF_CPSTOCF_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {

class Pass;

} // namespace mlir

namespace hail::ir {

/// Creates a pass to convert continuation-based control flow to CFG
/// branch-based operation in the ControlFlow dialect.
auto createCPSToCFPass() -> std::unique_ptr<mlir::Pass>;

} // namespace hail::ir

#endif // HAIL_CONVERSION_CPSTOCF_CPSTOCF_H
