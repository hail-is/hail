#ifndef CONVERSION_LOWEROPTION_LOWEROPTION_H
#define CONVERSION_LOWEROPTION_LOWEROPTION_H

#include <memory>

namespace mlir {

class Pass;
class RewritePatternSet;


} // namespace mlir

namespace hail::ir {

void populateLowerOptionConversionPatterns(mlir::RewritePatternSet &patterns);

/// Creates a pass to lower the Option type to bool and values ssa-values, using CPS for control
/// flow
std::unique_ptr<mlir::Pass> createLowerOptionPass();

} // namespace hail::ir

#endif // CONVERSION_LOWEROPTION_LOWEROPTION_H