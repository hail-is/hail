#include "../PassDetail.h"
#include "Conversion/LowerSandbox/LowerSandbox.h"
#include "Dialect/CPS/IR/CPS.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

#include <vector>

namespace hail::ir {

struct CPSToCFPass : public CPSToCFBase<CPSToCFPass> {
  void runOnOperation() override;

private:
  mlir::LogicalResult lowerCallCCOp(mlir::IRRewriter &rewriter, CallCCOp op);
  mlir::LogicalResult lowerDefContOp(mlir::IRRewriter &rewriter, DefContOp op);
};

mlir::LogicalResult CPSToCFPass::lowerCallCCOp(mlir::IRRewriter &rewriter, CallCCOp op) {
  // Split the current block before the CallCCOp to create the continuation point.
  mlir::Block *parentBlock = op->getBlock();
  mlir::Block *continuation = rewriter.splitBlock(parentBlock, op->getIterator());

  // add args to continuation block for each return value of op
  llvm::SmallVector<mlir::Location, 4> locs(op->getNumResults(), op.getLoc());
  continuation->addArguments(op->getResultTypes(), locs);

  // Replace all uses of captured continuation (which must be ApplyContOps) with branches to the
  // continuation block. 'make_early_inc_range' finds the next user before executing the body, so
  // that deleting the current user is safe
  mlir::Region &body = op.body();
  for (mlir::OpOperand &use : llvm::make_early_inc_range(body.getArgument(0).getUses())) {
    auto applyOp = dyn_cast<ApplyContOp>(use.getOwner());
    // continuations can only be used as first operand to an apply op
    if ((applyOp == nullptr) || (use.getOperandNumber() != 0)) {
      signalPassFailure();
      return mlir::failure();
    }
    rewriter.setInsertionPoint(applyOp);
    rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(applyOp, applyOp.args(), continuation);
  }
  assert(body.getArgument(0).use_empty() && "should be no uses of callcc body arg");
  // continuation arg now has no uses, safe to delete
  body.eraseArgument(0);
  assert(body.getNumArguments() == 0);

  mlir::Block &bodyEntryBlock = body.front();
  // insert the (now fully lowered) body into the parent region
  rewriter.inlineRegionBefore(body, continuation);
  assert(body.empty());
  // append the callcc body to the end of parentBlock
  rewriter.mergeBlocks(&bodyEntryBlock, parentBlock);

  // finally, replace results of callcc with continuation args
  rewriter.replaceOp(op, continuation->getArguments());
  return mlir::success();
}

mlir::LogicalResult CPSToCFPass::lowerDefContOp(mlir::IRRewriter &rewriter, DefContOp op) {
  mlir::Block &body = op.bodyRegion().front();
  // inline body into parent region
  rewriter.inlineRegionBefore(op.bodyRegion(), *op->getParentRegion(),
                              std::next(op->getBlock()->getIterator()));
  // replace all uses (which must be ApplyContOps) with branches to the body block
  // make_early_inc_range finds the next user before executing the body, so that deleting the
  // current user is safe
  for (mlir::OpOperand &use : llvm::make_early_inc_range(op.result().getUses())) {
    auto applyOp = dyn_cast<ApplyContOp>(use.getOwner());
    // continuations can only be used as first operand to an apply op
    if ((applyOp == nullptr) || (use.getOperandNumber() != 0)) {
      signalPassFailure();
      return mlir::failure();
    }
    rewriter.setInsertionPoint(applyOp);
    assert(body.getNumArguments() == applyOp.args().size());
    rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(applyOp, applyOp.args(), &body);
  }
  assert(op.result().use_empty() && "should be no uses of defcont result");
  // there are no more uses, safe to delete
  rewriter.eraseOp(op);
  return mlir::success();
}

void CPSToCFPass::runOnOperation() {
  std::vector<mlir::Operation *> worklist;
  worklist.reserve(64);

  // add nested ops to worklist in postorder
  auto *root = getOperation();
  for (auto &region : getOperation()->getRegions())
    region.walk([&worklist](mlir::Operation *op) {
      if (isa<CallCCOp>(op) || isa<DefContOp>(op))
        worklist.push_back(op);
    });

  mlir::IRRewriter rewriter(root->getContext());
  for (auto *op : worklist) {
    op->emitWarning() << "Visiting:";
    if (auto callcc = dyn_cast<CallCCOp>(op)) {
      if (lowerCallCCOp(rewriter, callcc).failed())
        return;
    } else if (auto defcont = dyn_cast<DefContOp>(op)) {
      if (lowerDefContOp(rewriter, defcont).failed())
        return;
    }
    root->emitWarning() << "After change:";
  }
}

std::unique_ptr<mlir::Pass> createCPSToCFPass() { return std::make_unique<CPSToCFPass>(); }

} // namespace hail::ir