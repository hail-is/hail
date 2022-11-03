#include "Conversion/LowerSandbox/LowerSandbox.h"
#include "Dialect/Sandbox/IR/Sandbox.h"
#include "../PassDetail.h"

#include "mlir/Transforms/DialectConversion.h"

namespace hail {

struct LowerSandboxPass
    : public LowerSandboxBase<LowerSandboxPass> {
  void runOnOperation() override;
};

void LowerSandboxPass::runOnOperation() {
  mlir::RewritePatternSet patterns(&getContext());
  populateLowerSandboxConversionPatterns(patterns);

  // Configure conversion to lower out SCF operations.
  mlir::ConversionTarget target(getContext());
  target.addIllegalOp<ir::ConstantOp, ir::AddIOp>();
  target.markUnknownOpDynamicallyLegal([](mlir::Operation *) { return true; });
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

void populateLowerSandboxConversionPatterns(
    mlir::RewritePatternSet &patterns) {
  // patterns.add<>(patterns.getContext());
}

std::unique_ptr<mlir::Pass> createLowerSandboxPass() {
  return std::make_unique<LowerSandboxPass>();
}

} // namespace hail