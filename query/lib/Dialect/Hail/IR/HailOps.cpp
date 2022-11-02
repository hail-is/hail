#include "Dialect/Hail/IR/HailDialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Location.h"
#include <cstddef>

#define GET_OP_CLASSES
#include "Dialect/Hail/IR/HailOps.cpp.inc"

namespace hail {
namespace ir{

mlir::OpFoldResult I32Op::fold(llvm::ArrayRef<mlir::Attribute> operands) {
  return valueAttr();
}

mlir::OpFoldResult I32Plus::fold(llvm::ArrayRef<mlir::Attribute> operands) {
  assert(operands.size() == 2 && "binary op takes two operands");
  if (!operands[0] || !operands[1])
    return {};

  if (operands[0].isa<mlir::IntegerAttr>() && operands[1].isa<mlir::IntegerAttr>()) {
    auto lhs = operands[0].cast<mlir::IntegerAttr>();
    auto rhs = operands[1].cast<mlir::IntegerAttr>();

    auto result = mlir::IntegerAttr::get(lhs.getType(), lhs.getValue() + rhs.getValue());
    assert(result);
    return result;
  }

  return {};
}

struct SimplifyAddConstAddConst : public mlir::OpRewritePattern<I32Plus> {
  SimplifyAddConstAddConst(mlir::MLIRContext *context)
      : OpRewritePattern<I32Plus>(context, /*benefit=*/1) {}

  mlir::LogicalResult matchAndRewrite(I32Plus op, mlir::PatternRewriter &rewriter) const override {
    auto lhs = op.left().getDefiningOp<I32Plus>();
    if (!lhs) return mlir::failure();
    
    auto lConst = lhs.right().getDefiningOp<I32Op>();
    auto rConst = op.right().getDefiningOp<I32Op>();
    if (!lConst || !rConst) return mlir::failure();

    auto sumConst = rewriter.create<I32Op>(
      mlir::FusedLoc::get(op->getContext(), {lConst->getLoc(), rConst.getLoc()}, nullptr),
      lConst.getType(), lConst.value() + rConst.value());
    rewriter.replaceOpWithNewOp<I32Plus>(op, op.output().getType(), lhs.left(), sumConst);
    return mlir::success();
  }
};

void I32Plus::getCanonicalizationPatterns(mlir::RewritePatternSet &patterns,
                                          mlir::MLIRContext *context) {
  patterns.add<SimplifyAddConstAddConst>(context);
}

} // end ir
} // end hail
