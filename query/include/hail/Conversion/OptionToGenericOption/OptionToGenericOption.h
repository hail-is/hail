#ifndef CONVERSION_OPTIONTOGENERICOPTION_OPTIONTOGENERICOPTION_H
#define CONVERSION_OPTIONTOGENERICOPTION_OPTIONTOGENERICOPTION_H

#include <memory>

namespace mlir {

class Pass;
class RewritePatternSet;

} // namespace mlir

namespace hail::ir {

void populateOptionToGenericOptionConversionPatterns(mlir::RewritePatternSet &patterns);

std::unique_ptr<mlir::Pass> createOptionToGenericOptionPass();

} // namespace hail::ir

#endif // CONVERSION_OPTIONTOGENERICOPTION_OPTIONTOGENERICOPTION_H
