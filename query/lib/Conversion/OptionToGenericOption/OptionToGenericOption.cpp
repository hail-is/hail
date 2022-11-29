#include "Conversion/OptionToGenericOption/OptionToGenericOption.h"
#include "../PassDetail.h"
#include "Dialect/CPS/IR/CPS.h"
#include "Dialect/Option/IR/Option.h"

#include "mlir/Transforms/DialectConversion.h"

namespace hail::ir {

struct OptionToGenericOptionPass : public OptionToGenericOptionBase<OptionToGenericOptionPass> {
  void runOnOperation() override;
};

namespace {

class ConvertMapOp : public mlir::OpConversionPattern<MapOp> {
public:
  explicit ConvertMapOp(mlir::MLIRContext *context)
      : OpConversionPattern<MapOp>(context, /*benefit=*/1) {}

  mlir::LogicalResult matchAndRewrite(MapOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    llvm::SmallVector<mlir::Type> valueTypes;
    for (auto option : op.getOperandTypes()) {
      auto types = option.cast<OptionType>().getValueTypes();
      valueTypes.append(types.begin(), types.end());
    }

    auto construct = rewriter.create<ConstructOp>(loc, op.getType().getValueTypes());

    rewriter.setInsertionPointToStart(construct.getBody());
    auto bodyCont = rewriter.create<DefContOp>(loc, valueTypes);
    rewriter.mergeBlocks(op.getBody(), bodyCont.getBody(), bodyCont.getBody()->getArguments());
    auto yield = llvm::cast<YieldOp>(bodyCont.getBody()->getTerminator());
    rewriter.setInsertionPointAfter(yield);
    rewriter.replaceOpWithNewOp<ApplyContOp>(yield, construct.getPresentCont(),
                                             yield.getOperands());

    rewriter.setInsertionPointAfter(bodyCont);
    rewriter.create<DestructOp>(loc, op.getOperands(), construct.getMissingCont(), bodyCont);

    rewriter.replaceOp(op, construct->getResults());
    return mlir::success();
  }
};

} // end namespace

void populateOptionToGenericOptionConversionPatterns(mlir::RewritePatternSet &patterns) {
  patterns.add<ConvertMapOp>(patterns.getContext());
}

void OptionToGenericOptionPass::runOnOperation() {
  mlir::RewritePatternSet patterns(&getContext());
  populateOptionToGenericOptionConversionPatterns(patterns);

  mlir::ConversionTarget target(getContext());
  target.addIllegalDialect<OptionDialect>();
  target.addLegalOp<ConstructOp, DestructOp>();

  // We want to completely lower to LLVM, so we use a `FullConversion`. This
  // ensures that only legal operations will remain after the conversion.
  target.markUnknownOpDynamicallyLegal([](mlir::Operation *) { return true; });
  if (failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> createOptionToGenericOptionPass() {
  return std::make_unique<OptionToGenericOptionPass>();
}

} // namespace hail::ir