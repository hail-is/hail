#include "hail/Conversion/LowerOption/LowerOption.h"

#include "../PassDetail.h"

#include "hail/Dialect/CPS/IR/CPS.h"
#include "hail/Dialect/Option/IR/Option.h"
#include "hail/Support/MLIR.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"

#include <vector>

namespace hail::ir {

namespace {

struct LowerOptionPass : public LowerOptionBase<LowerOptionPass> {
  void runOnOperation() override;
};

struct LoweredOption {
  explicit LoweredOption(OptionType type) { operands.reserve(type.getValueTypes().size() + 1); };
  auto isDefined() -> Value { return operands[0]; };
  auto values() -> ValueRange { return ValueRange(operands).drop_front(1); };

  SmallVector<Value, 4> operands{};
};

auto unpackOptional(ConversionPatternRewriter &rewriter, mlir::Location loc, Value optional)
    -> LoweredOption {
  auto type = optional.getType().cast<OptionType>();
  SmallVector<mlir::Type, 2> resultTypes;
  resultTypes.reserve(type.getValueTypes().size() + 1);
  LoweredOption result{type};
  resultTypes.push_back(rewriter.getI1Type());
  resultTypes.append(type.getValueTypes().begin(), type.getValueTypes().end());
  rewriter.createOrFold<mlir::UnrealizedConversionCastOp>(result.operands, loc, resultTypes,
                                                          optional);
  return result;
}

struct ConstructOpConversion : public OpConversionPattern<ConstructOp> {
  ConstructOpConversion(MLIRContext *context)
      : OpConversionPattern<ConstructOp>(context, /*benefit=*/1) {}

  auto matchAndRewrite(ConstructOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const
      -> LogicalResult override {
    auto loc = op.getLoc();
    SmallVector<Type, 4> resultTypes;
    auto valueTypes = op.getType().getValueTypes();
    resultTypes.reserve(valueTypes.size() + 1);
    resultTypes.push_back(rewriter.getI1Type());
    resultTypes.append(valueTypes.begin(), valueTypes.end());

    auto callcc = rewriter.create<CallCCOp>(loc, resultTypes);
    Value const retCont = callcc.body().getArgument(0);

    SmallVector<Value, 4> results;
    results.reserve(valueTypes.size() + 1);
    auto &body = op.bodyRegion();

    // Define the new missing continuation
    rewriter.setInsertionPointToStart(&callcc.body().front());
    auto missingCont = rewriter.create<DefContOp>(loc);
    rewriter.setInsertionPointToStart(&missingCont.bodyRegion().front());
    auto constFalse = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getBoolAttr(false));
    results.push_back(constFalse);
    llvm::transform(valueTypes, std::back_inserter(results),
                    [&](mlir::Type type) { return rewriter.create<UndefinedOp>(loc, type); });
    rewriter.create<ApplyContOp>(loc, retCont, results);

    // Define the new present continuation
    results.clear();
    rewriter.setInsertionPointAfter(missingCont);
    auto presentCont = rewriter.create<DefContOp>(loc, valueTypes);
    rewriter.setInsertionPointToStart(&presentCont.bodyRegion().front());
    auto constTrue = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getBoolAttr(true));
    results.push_back(constTrue);
    results.append(presentCont.bodyRegion().args_begin(), presentCont.bodyRegion().args_end());
    rewriter.create<ApplyContOp>(loc, retCont, results);

    rewriter.mergeBlocks(&body.front(), &callcc.body().front(), {missingCont, presentCont});

    // Cast results back to Option type and replace
    rewriter.setInsertionPointAfter(callcc);
    rewriter.replaceOpWithNewOp<mlir::UnrealizedConversionCastOp>(op, op.getType(),
                                                                  callcc.getResults());

    return mlir::success();
  }
};

struct DestructOpConversion : public OpConversionPattern<DestructOp> {
  using OpConversionPattern<DestructOp>::OpConversionPattern;

  auto matchAndRewrite(DestructOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const
      -> LogicalResult override {
    SmallVector<Value> values;
    Value isDefined;
    for (auto option : adaptor.inputs()) {
      LoweredOption unpack = unpackOptional(rewriter, op.getLoc(), option);
      values.append(unpack.values().begin(), unpack.values().end());
      if (!isDefined) {
        isDefined = unpack.isDefined();
      } else {
        isDefined =
            rewriter.create<mlir::arith::AndIOp>(op.getLoc(), isDefined, unpack.isDefined());
      }
    }

    rewriter.replaceOpWithNewOp<IfOp>(op, isDefined, op.presentCont(), values, op.missingCont(),
                                      ValueRange{});
    return mlir::success();
  }
};

} // end namespace

void LowerOptionPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateLowerOptionConversionPatterns(patterns);

  ConversionTarget target(getContext());
  target.addIllegalDialect<OptionDialect>();
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

void populateLowerOptionConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<ConstructOpConversion, DestructOpConversion>(patterns.getContext());
}

auto createLowerOptionPass() -> std::unique_ptr<mlir::Pass> {
  return std::make_unique<LowerOptionPass>();
}

} // namespace hail::ir
