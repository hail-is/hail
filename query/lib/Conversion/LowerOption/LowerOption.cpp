#include "Conversion/LowerOption/LowerOption.h"
#include "../PassDetail.h"
#include "Dialect/CPS/IR/CPS.h"
#include "Dialect/Option/IR/Option.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

#include <vector>

namespace hail::ir {

namespace {

struct LowerOptionPass : public LowerOptionBase<LowerOptionPass> {
  void runOnOperation() override;
};

mlir::Value packOptional(mlir::PatternRewriter &rewriter, mlir::Location loc, OptionType type,
                         mlir::Value isPresent, mlir::ValueRange values) {
  assert(type.getValueTypes() == values.getTypes());
  llvm::SmallVector<mlir::Value, 1> results;
  llvm::SmallVector<mlir::Value, 2> toCast;
  toCast.reserve(values.size() + 1);
  toCast.push_back(isPresent);
  toCast.append(values.begin(), values.end());
  rewriter.createOrFold<mlir::UnrealizedConversionCastOp>(results, loc, type, toCast);
  return results[0];
}

struct LoweredOption {
  explicit LoweredOption(OptionType type) { operands.reserve(type.getValueTypes().size() + 1); };
  mlir::Value isDefined() { return operands[0]; };
  mlir::ValueRange values() { return mlir::ValueRange(operands).drop_front(1); };

  llvm::SmallVector<mlir::Value, 4> operands{};
};

LoweredOption unpackOptional(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
                             mlir::Value optional) {
  auto type = optional.getType().cast<OptionType>();
  llvm::SmallVector<mlir::Type, 2> resultTypes;
  resultTypes.reserve(type.getValueTypes().size() + 1);
  LoweredOption result{type};
  resultTypes.push_back(rewriter.getI1Type());
  resultTypes.append(type.getValueTypes().begin(), type.getValueTypes().end());
  rewriter.createOrFold<mlir::UnrealizedConversionCastOp>(result.operands, loc, resultTypes,
                                                          optional);
  return result;
}

struct ConstructOpConversion : public mlir::OpConversionPattern<ConstructOp> {
  ConstructOpConversion(mlir::MLIRContext *context)
      : OpConversionPattern<ConstructOp>(context, /*benefit=*/1) {}

  mlir::LogicalResult matchAndRewrite(ConstructOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    llvm::SmallVector<mlir::Type, 4> resultTypes;
    auto valueTypes = op.getType().getValueTypes();
    resultTypes.reserve(valueTypes.size() + 1);
    resultTypes.push_back(rewriter.getI1Type());
    resultTypes.append(valueTypes.begin(), valueTypes.end());

    auto callcc = rewriter.create<CallCCOp>(loc, resultTypes);
    callcc.body().emplaceBlock();
    callcc.body().addArgument(rewriter.getType<ContinuationType>(resultTypes), loc);
    mlir::Value retCont = callcc.body().getArgument(0);

    llvm::SmallVector<mlir::Value, 4> results;
    results.reserve(valueTypes.size() + 1);
    auto &body = op.bodyRegion();

    // Define the new missing continuation
    rewriter.setInsertionPointToStart(&callcc.body().front());
    auto missingCont = rewriter.create<DefContOp>(
        loc, rewriter.getType<ContinuationType>(llvm::ArrayRef<mlir::Type>()));
    missingCont.bodyRegion().emplaceBlock();
    rewriter.setInsertionPointToStart(&missingCont.bodyRegion().front());
    auto constFalse = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getBoolAttr(false));
    results.push_back(constFalse);
    llvm::transform(valueTypes, std::back_inserter(results),
                    [&](mlir::Type type) { return rewriter.create<UndefinedOp>(loc, type); });
    rewriter.create<ApplyContOp>(loc, retCont, results);

    // Define the new present continuation
    results.clear();
    rewriter.setInsertionPointAfter(missingCont);
    auto presentCont =
        rewriter.create<DefContOp>(loc, rewriter.getType<ContinuationType>(valueTypes));
    presentCont.bodyRegion().emplaceBlock();
    llvm::SmallVector<mlir::Location, 4> locs(valueTypes.size(), loc);
    presentCont.bodyRegion().addArguments(valueTypes, locs);
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

struct DestructOpConversion : public mlir::OpConversionPattern<DestructOp> {
  using OpConversionPattern<DestructOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(DestructOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Value> values;
    mlir::Value isDefined;
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
                                      mlir::ValueRange{});
    return mlir::success();
  }
};

} // end namespace

void LowerOptionPass::runOnOperation() {
  mlir::RewritePatternSet patterns(&getContext());
  populateLowerOptionConversionPatterns(patterns);

  mlir::ConversionTarget target(getContext());
  target.addIllegalDialect<OptionDialect>();
  target.markUnknownOpDynamicallyLegal([](mlir::Operation *) { return true; });
  if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

void populateLowerOptionConversionPatterns(mlir::RewritePatternSet &patterns) {
  patterns.add<ConstructOpConversion, DestructOpConversion>(patterns.getContext());
}

std::unique_ptr<mlir::Pass> createLowerOptionPass() { return std::make_unique<LowerOptionPass>(); }

} // namespace hail::ir