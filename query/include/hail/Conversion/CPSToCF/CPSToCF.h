#ifndef CONVERSION_CPSTOCF_CPSTOCF_H
#define CONVERSION_CPSTOCF_CPSTOCF_H

#include <memory>

namespace mlir {

class Pass;

} // namespace mlir

namespace hail::ir {

/// Creates a pass to convert continuation-based control flow to CFG
/// branch-based operation in the ControlFlow dialect.
std::unique_ptr<mlir::Pass> createCPSToCFPass();

} // namespace hail::ir

#endif // CONVERSION_CPSTOCF_CPSTOCF_H
