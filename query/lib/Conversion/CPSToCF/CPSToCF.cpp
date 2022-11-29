#include "../PassDetail.h"
#include "Conversion/LowerSandbox/LowerSandbox.h"
#include "Dialect/CPS/IR/CPS.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

#include <vector>

namespace hail::ir {

struct CPSToCFPass : public CPSToCFBase<CPSToCFPass> {
  void runOnOperation() override;
};

} // namespace hail::ir

using namespace hail::ir;

namespace {

void lowerCallCCOp(mlir::IRRewriter &rewriter, CallCCOp callcc, std::vector<DefContOp> &defsWorklist) {
  auto loc = callcc->getLoc();
  assert(callcc->getParentRegion()->hasOneBlock());
  // Split the current block before callcc to create the continuation point.
  mlir::Block *parentBlock = callcc->getBlock();
  mlir::Block *continuation = rewriter.splitBlock(parentBlock, callcc->getIterator());

  // create a DefContOp holding the continuation of callcc
  rewriter.setInsertionPointToEnd(parentBlock);
  auto defcont = rewriter.create<DefContOp>(loc, callcc->getResultTypes());
  rewriter.mergeBlocks(continuation, defcont.getBody(), {});

  defsWorklist.push_back(defcont);

  // inline the body of callcc, replacing the return continuation with the defcont,
  // and replacing uses of the results with the args of the defcont
  rewriter.mergeBlocks(callcc.getBody(), parentBlock, defcont.getResult());
  rewriter.replaceOp(callcc, defcont.getBody()->getArguments());
  return;
}

mlir::Block *getDefBlock(mlir::Value cont) {
  auto def = cont.getDefiningOp<DefContOp>();
  assert(def && "Continuation def is not visable");
  return &def.bodyRegion().front();
}

void lowerApplyContOp(mlir::IRRewriter &rewriter, ApplyContOp op) {
  rewriter.setInsertionPoint(op);
  rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(op, op.args(), getDefBlock(op.cont()));
}

void lowerIfOp(mlir::IRRewriter &rewriter, IfOp op) {
  rewriter.setInsertionPoint(op);
  rewriter.replaceOpWithNewOp<mlir::cf::CondBranchOp>(op, op.condition(),
                                                      getDefBlock(op.trueCont()), op.trueArgs(),
                                                      getDefBlock(op.falseCont()), op.falseArgs());
}

void lowerDefContOp(mlir::IRRewriter &rewriter, DefContOp op) {
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
  defsWorklist.reserve(64);
  usesWorklist.reserve(64);

  // add nested ops to worklist in postorder
  auto *root = getOperation();
  for (auto &region : getOperation()->getRegions())
    region.walk([&](mlir::Operation *op) {
      if (auto callcc = dyn_cast<CallCCOp>(op))
        callccsWorklist.push_back(callcc);
      else if (auto defcont = dyn_cast<DefContOp>(op))
        defsWorklist.push_back(defcont);
      else if (isa<ApplyContOp>(op) || isa<IfOp>(op))
        usesWorklist.push_back(op);
    });

  mlir::IRRewriter rewriter(root->getContext());
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

std::unique_ptr<mlir::Pass> createCPSToCFPass() { return std::make_unique<CPSToCFPass>(); }

} // namespace hail::ir