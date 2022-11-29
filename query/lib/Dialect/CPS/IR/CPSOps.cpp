#include "Dialect/CPS/IR/CPS.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

#define GET_OP_CLASSES
#include "Dialect/CPS/IR/CPSOps.cpp.inc"

using namespace hail::ir;

namespace {

struct InlineCont : public mlir::OpRewritePattern<ApplyContOp> {
  using OpRewritePattern<ApplyContOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(ApplyContOp apply,
                                      mlir::PatternRewriter &rewriter) const override {
    auto defcont = apply.cont().getDefiningOp<DefContOp>();
    if (!defcont || !defcont->hasOneUse())
      return mlir::failure();

    rewriter.mergeBlocks(defcont.getBody(), apply->getBlock(), apply.args());
    rewriter.eraseOp(apply);
    rewriter.eraseOp(defcont);

    return mlir::success();
  }
};

struct TrivialCallCC : public mlir::OpRewritePattern<CallCCOp> {
  using OpRewritePattern<CallCCOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(CallCCOp callcc,
                                      mlir::PatternRewriter &rewriter) const override {
    auto terminator = dyn_cast<ApplyContOp>(callcc.getBody()->getTerminator());
    if (!terminator)
      return mlir::failure();
    if (terminator.cont() != callcc.getBody()->getArgument(0))
      return mlir::failure();
    llvm::SmallVector<mlir::Value> values(terminator.args().begin(), terminator.args().end());
    rewriter.eraseOp(terminator);
    callcc.getBody()->eraseArgument(0);
    rewriter.mergeBlockBefore(callcc.getBody(), callcc);
    rewriter.replaceOp(callcc, values);
    return mlir::success();
  }
};

} // namespace

void ApplyContOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                              mlir::MLIRContext *context) {
  results.add<InlineCont>(context);
}

void CallCCOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                              mlir::MLIRContext *context) {
  results.add<TrivialCallCC>(context);
}

void CallCCOp::build(mlir::OpBuilder &builder, mlir::OperationState &result, mlir::TypeRange resultTypes) {
  result.addTypes(resultTypes);
  auto region = result.addRegion();
  region->emplaceBlock();
  region->addArgument(builder.getType<ContinuationType>(resultTypes), result.location);
}

void DefContOp::build(mlir::OpBuilder &builder, mlir::OperationState &result, mlir::TypeRange argTypes) {
  result.addTypes(builder.getType<ContinuationType>(argTypes));
  auto region = result.addRegion();
  region->emplaceBlock();
  llvm::SmallVector<mlir::Location> locs(argTypes.size(), result.location);
  region->addArguments(argTypes, locs);
}