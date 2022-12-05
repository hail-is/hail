#ifndef CONVERSION_LOWERSANDBOX_LOWERSANDBOX_H
#define CONVERSION_LOWERSANDBOX_LOWERSANDBOX_H

#include <memory>

namespace mlir {

class Pass;
class RewritePatternSet;

} // namespace mlir

namespace hail::ir {

/// Collect a set of patterns to convert SCF operations to CFG branch-based
/// operations within the ControlFlow dialect.
void populateLowerSandboxConversionPatterns(mlir::RewritePatternSet &patterns);

/// Creates a pass to convert SCF operations to CFG branch-based operation in
/// the ControlFlow dialect.
std::unique_ptr<mlir::Pass> createLowerSandboxPass();

} // namespace hail::ir

#endif // CONVERSION_LOWERSANDBOX_LOWERSANDBOX_H