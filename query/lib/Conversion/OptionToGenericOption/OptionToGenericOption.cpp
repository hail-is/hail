#include "hail/Conversion/OptionToGenericOption/OptionToGenericOption.h"

#include "../PassDetail.h"

#include "hail/Dialect/CPS/IR/CPS.h"
#include "hail/Dialect/Option/IR/Option.h"
#include "hail/Support/MLIR.h"

#include "mlir/Transforms/DialectConversion.h"

namespace hail::ir {

struct OptionToGenericOptionPass : public OptionToGenericOptionBase<OptionToGenericOptionPass> {
  void runOnOperation() override;
};

namespace {

class ConvertMapOp : public OpConversionPattern<MapOp> {
public:
  explicit ConvertMapOp(MLIRContext *context)
      : OpConversionPattern<MapOp>(context, /*benefit=*/1) {}

  auto matchAndRewrite(MapOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const
      -> LogicalResult override {
    auto loc = op.getLoc();

    SmallVector<Type> valueTypes;
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
    return success();
  }
};

} // end namespace

void populateOptionToGenericOptionConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertMapOp>(patterns.getContext());
}

void OptionToGenericOptionPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateOptionToGenericOptionConversionPatterns(patterns);

  ConversionTarget target(getContext());
  target.addIllegalDialect<OptionDialect>();
  target.addLegalOp<ConstructOp, DestructOp>();

  // We want to completely lower to LLVM, so we use a `FullConversion`. This
  // ensures that only legal operations will remain after the conversion.
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  if (failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

auto createOptionToGenericOptionPass() -> std::unique_ptr<Pass> {
  return std::make_unique<OptionToGenericOptionPass>();
}

} // namespace hail::ir
