#include "hail/Conversion/CPSToCF/CPSToCF.h"

#include "../PassDetail.h"

#include "hail/Dialect/CPS/IR/CPS.h"
#include "hail/Support/MLIR.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"

#include <vector>

namespace hail::ir {

struct CPSToCFPass : public CPSToCFBase<CPSToCFPass> {
  void runOnOperation() override;
};

} // namespace hail::ir

using namespace hail::ir;

namespace {

void lowerCallCCOp(IRRewriter &rewriter, CallCCOp callcc, std::vector<DefContOp> &defsWorklist) {
  auto loc = callcc->getLoc();
  assert(callcc->getParentRegion()->hasOneBlock());
  // Split the current block before callcc to create the continuation point.
  Block *parentBlock = callcc->getBlock();
  Block *continuation = rewriter.splitBlock(parentBlock, callcc->getIterator());

  // create a DefContOp holding the continuation of callcc
  rewriter.setInsertionPointToEnd(parentBlock);
  auto defcont = rewriter.create<DefContOp>(loc, callcc->getResultTypes());
  rewriter.mergeBlocks(continuation, defcont.getBody(), {});

  defsWorklist.push_back(defcont);

  // inline the body of callcc, replacing the return continuation with the defcont,
  // and replacing uses of the results with the args of the defcont
  rewriter.mergeBlocks(callcc.getBody(), parentBlock, defcont.getResult());
  rewriter.replaceOp(callcc, defcont.getBody()->getArguments());
}

auto getDefBlock(Value cont) -> Block * {
  auto def = cont.getDefiningOp<DefContOp>();
  assert(def && "Continuation def is not visable");
  return &def.bodyRegion().front();
}

void lowerApplyContOp(IRRewriter &rewriter, ApplyContOp op) {
  rewriter.setInsertionPoint(op);
  rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(op, op.args(), getDefBlock(op.cont()));
}

void lowerIfOp(IRRewriter &rewriter, IfOp op) {
  rewriter.setInsertionPoint(op);
  rewriter.replaceOpWithNewOp<mlir::cf::CondBranchOp>(op, op.condition(),
                                                      getDefBlock(op.trueCont()), op.trueArgs(),
                                                      getDefBlock(op.falseCont()), op.falseArgs());
}

void lowerDefContOp(IRRewriter &rewriter, DefContOp op) {
  rewriter.inlineRegionBefore(op.bodyRegion(), *op->getParentRegion(),
                              std::next(op->getBlock()->getIterator()));
  rewriter.eraseOp(op);
}

} // namespace

namespace hail::ir {

void CPSToCFPass::runOnOperation() {
  std::vector<CallCCOp> callccsWorklist;
  std::vector<DefContOp> defsWorklist;
  std::vector<mlir::Operation *> usesWorklist;
  constexpr int initCapacity = 64;
  defsWorklist.reserve(initCapacity);
  usesWorklist.reserve(initCapacity);

  // add nested ops to worklist in postorder
  auto *root = getOperation();
  for (auto &region : getOperation()->getRegions())
    region.walk([&](Operation *op) {
      if (auto callcc = dyn_cast<CallCCOp>(op))
        callccsWorklist.push_back(callcc);
      else if (auto defcont = dyn_cast<DefContOp>(op))
        defsWorklist.push_back(defcont);
      else if (isa<ApplyContOp>(op) || isa<IfOp>(op))
        usesWorklist.push_back(op);
    });

  IRRewriter rewriter(root->getContext());
  for (auto callcc : callccsWorklist) {
    lowerCallCCOp(rewriter, callcc, defsWorklist);
  }
  for (auto *op : usesWorklist) {
    if (auto apply = dyn_cast<ApplyContOp>(op)) {
      lowerApplyContOp(rewriter, apply);
    } else if (auto ifOp = dyn_cast<IfOp>(op)) {
      lowerIfOp(rewriter, ifOp);
    }
  }
  for (auto defcont : defsWorklist) {
    lowerDefContOp(rewriter, defcont);
  }
}

auto createCPSToCFPass() -> std::unique_ptr<Pass> { return std::make_unique<CPSToCFPass>(); }

} // namespace hail::ir
