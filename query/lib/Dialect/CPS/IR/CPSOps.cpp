#include "hail/Dialect/CPS/IR/CPS.h"

#include "hail/Support/MLIR.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

#define GET_OP_CLASSES
#include "hail/Dialect/CPS/IR/CPSOps.cpp.inc"

using namespace hail::ir;

//===----------------------------------------------------------------------===//
// ApplyContOp
//===----------------------------------------------------------------------===//

namespace {
struct InlineCont : public OpRewritePattern<ApplyContOp> {
  using OpRewritePattern<ApplyContOp>::OpRewritePattern;

  auto matchAndRewrite(ApplyContOp apply, PatternRewriter &rewriter) const
      -> LogicalResult override {
    auto defcont = apply.cont().getDefiningOp<DefContOp>();
    if (!defcont || !defcont->hasOneUse())
      return failure();

    rewriter.mergeBlocks(defcont.getBody(), apply->getBlock(), apply.args());
    rewriter.eraseOp(apply);
    rewriter.eraseOp(defcont);

    return success();
  }
};

} // namespace

void ApplyContOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) {
  results.add<InlineCont>(context);
}

//===----------------------------------------------------------------------===//
// CallCCOp
//===----------------------------------------------------------------------===//

void CallCCOp::build(OpBuilder &odsBuilder, OperationState &odsState, TypeRange resultTypes) {
  odsState.addTypes(resultTypes);
  auto *region = odsState.addRegion();
  region->emplaceBlock();
  region->addArgument(odsBuilder.getType<ContinuationType>(resultTypes), odsState.location);
}

namespace {

struct TrivialCallCC : public OpRewritePattern<CallCCOp> {
  using OpRewritePattern<CallCCOp>::OpRewritePattern;

  auto matchAndRewrite(CallCCOp callcc, PatternRewriter &rewriter) const -> LogicalResult override {
    auto terminator = dyn_cast<ApplyContOp>(callcc.getBody()->getTerminator());
    if (!terminator)
      return failure();
    if (terminator.cont() != callcc.getBody()->getArgument(0))
      return failure();
    SmallVector<Value> const values(terminator.args().begin(), terminator.args().end());
    rewriter.eraseOp(terminator);
    callcc.getBody()->eraseArgument(0);
    rewriter.mergeBlockBefore(callcc.getBody(), callcc);
    rewriter.replaceOp(callcc, values);
    return success();
  }
};

} // namespace

void CallCCOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) {
  results.add<TrivialCallCC>(context);
}

//===----------------------------------------------------------------------===//
// DefContOp
//===----------------------------------------------------------------------===//

void DefContOp::build(OpBuilder &odsBuilder, OperationState &odsState, TypeRange argTypes) {
  odsState.addTypes(odsBuilder.getType<ContinuationType>(argTypes));
  auto *region = odsState.addRegion();
  region->emplaceBlock();
  SmallVector<Location> const locs(argTypes.size(), odsState.location);
  region->addArguments(argTypes, locs);
}

auto DefContOp::verifyRegions() -> LogicalResult {
  auto *body = getBody();
  ContinuationType const type = getType();
  auto numArgs = type.getInputs().size();

  if (body->getNumArguments() != numArgs)
    return emitOpError("mismatch in number of basic block args and continuation type");

  unsigned i = 0;
  for (auto e : llvm::zip(body->getArguments(), type.getInputs())) {
    if (std::get<0>(e).getType() != std::get<1>(e))
      return emitOpError() << "type mismatch between " << i << "th block arg and continuation type";

    i++;
  }
  return success();
}
