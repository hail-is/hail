#include "Dialect/Option/IR/Option.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

#define GET_OP_CLASSES
#include "Dialect/Option/IR/OptionOps.cpp.inc"

using namespace hail::ir;

//===----------------------------------------------------------------------===//
// ConstructOp
//===----------------------------------------------------------------------===//

// mlir::ParseResult ConstructOp::parse(mlir::OpAsmParser &parser,
//                                      mlir::OperationState &result) {
//   return mlir::success();
// }

// void ConstructOp::print(mlir::OpAsmPrinter &p) {
// }

//===----------------------------------------------------------------------===//
// DestructOp
//===----------------------------------------------------------------------===//

// mlir::ParseResult DestructOp::parse(mlir::OpAsmParser &parser,
//                                    mlir::OperationState &result) {
//   return mlir::success();
// }

// void DestructOp::print(mlir::OpAsmPrinter &p) {
// }

namespace {

struct DestructOfConstruct : public mlir::OpRewritePattern<DestructOp> {
  using OpRewritePattern<DestructOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(DestructOp destruct,
                                      mlir::PatternRewriter &rewriter) const override {
    if (auto construct = destruct.input().getDefiningOp<ConstructOp>()) {
      if (!construct->hasOneUse()) {
        return mlir::failure();
      }
      // llvm::errs() << "construct has " << construct->getBlock()->getNumArguments() << " block args";
      // llvm::errs() << "slice has size " << destruct.getOperands().slice(1, 2).size();
      rewriter.mergeBlockBefore(&construct.bodyRegion().front(), destruct,
                                destruct.getOperands().slice(1, 2));
      rewriter.eraseOp(destruct);
      rewriter.eraseOp(construct);
      return mlir::success();
    }

    return mlir::failure();
  }
};

} // end namespace

void DestructOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
  results.add<DestructOfConstruct>(context);
}
