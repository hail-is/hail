#include "Conversion/LowerSandbox/LowerSandbox.h"
#include "Dialect/Sandbox/IR/Sandbox.h"
#include "../PassDetail.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

namespace hail {

struct LowerSandboxPass
    : public LowerSandboxBase<LowerSandboxPass> {
  void runOnOperation() override;
};

namespace {

mlir::Value castToMLIR(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc, mlir::Value value) {
  auto castOp = rewriter.create<mlir::UnrealizedConversionCastOp>(
    loc,
     mlir::TypeRange{rewriter.getType<mlir::IntegerType>(32)},
      mlir::ValueRange{value});
  return castOp.getResult(0);
}
struct AddIOpConversion : public mlir::OpConversionPattern<ir::AddIOp> {
  AddIOpConversion(mlir::MLIRContext *context)
      : OpConversionPattern<ir::AddIOp>(context, /*benefit=*/1) {}

  mlir::LogicalResult
  matchAndRewrite(ir::AddIOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto lhs = castToMLIR(rewriter, op->getLoc(), adaptor.lhs());
    auto rhs = castToMLIR(rewriter, op->getLoc(), adaptor.rhs());
    rewriter.replaceOpWithNewOp<mlir::arith::AddIOp>(op, lhs, rhs);
    return mlir::success();
  }
};

struct ConstantOpConversion : public mlir::OpConversionPattern<ir::ConstantOp> {
  ConstantOpConversion(mlir::MLIRContext *context)
      : OpConversionPattern<ir::ConstantOp>(context, /*benefit=*/1) {}

  mlir::LogicalResult
  matchAndRewrite(ir::ConstantOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, adaptor.valueAttr(), rewriter.getType<mlir::IntegerType>(32));
    return mlir::success();
  }
};

} // end namespace

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
  patterns.add<ConstantOpConversion, AddIOpConversion>(patterns.getContext());
}

std::unique_ptr<mlir::Pass> createLowerSandboxPass() {
  return std::make_unique<LowerSandboxPass>();
}

} // namespace hail