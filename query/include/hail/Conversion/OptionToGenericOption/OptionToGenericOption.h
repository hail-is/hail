#ifndef HAIL_CONVERSION_OPTIONTOGENERICOPTION_OPTIONTOGENERICOPTION_H
#define HAIL_CONVERSION_OPTIONTOGENERICOPTION_OPTIONTOGENERICOPTION_H

#include <memory>

namespace mlir {

class Pass;
class RewritePatternSet;

} // namespace mlir

namespace hail::ir {

void populateOptionToGenericOptionConversionPatterns(mlir::RewritePatternSet &patterns);

auto createOptionToGenericOptionPass() -> std::unique_ptr<mlir::Pass>;

} // namespace hail::ir

#endif // HAIL_CONVERSION_OPTIONTOGENERICOPTION_OPTIONTOGENERICOPTION_H
