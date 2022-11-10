#include "Dialect/Sandbox/IR/Sandbox.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/APSInt.h"
#include <cstddef>

#include "Dialect/Sandbox/IR/SandboxOpsEnums.cpp.inc"

#define GET_OP_CLASSES

#include "Dialect/Sandbox/IR/SandboxOps.cpp.inc"
namespace hail {
namespace ir {

mlir::OpFoldResult ConstantOp::fold(llvm::ArrayRef<mlir::Attribute> operands) {
  return valueAttr();
}

mlir::LogicalResult ConstantOp::verify() {
  auto type = getType();
  // The value's type must match the return type.
  auto valueType = value().getType();

  if (valueType.isa<mlir::IntegerType>() && type.isa<IntType>()) {
    return mlir::success();
  }

  if (valueType == mlir::IntegerType::get(getContext(), 1) &&
      type.isa<BooleanType>()) {
    return mlir::success();
  }

  return emitOpError() << "bad constant: type=" << type
                       << ", valueType=" << valueType;
}

mlir::OpFoldResult AddIOp::fold(llvm::ArrayRef<mlir::Attribute> operands) {
  assert(operands.size() == 2 && "binary op takes two operands");
  if (!operands[0] || !operands[1])
    return {};

  if (operands[0].isa<mlir::IntegerAttr>() &&
      operands[1].isa<mlir::IntegerAttr>()) {
    auto lhs = operands[0].cast<mlir::IntegerAttr>();
    auto rhs = operands[1].cast<mlir::IntegerAttr>();

    auto result =
        mlir::IntegerAttr::get(lhs.getType(), lhs.getValue() + rhs.getValue());
    assert(result);
    return result;
  }

  return {};
}

mlir::OpFoldResult
ComparisonOp::fold(llvm::ArrayRef<mlir::Attribute> operands) {
  assert(operands.size() == 2 && "comparison op takes two operands");
  if (!operands[0] || !operands[1])
    return {};

  if (operands[0].isa<mlir::IntegerAttr>() &&
      operands[1].isa<mlir::IntegerAttr>()) {
    auto pred = predicate();
    auto lhs = operands[0].cast<mlir::IntegerAttr>();
    auto rhs = operands[1].cast<mlir::IntegerAttr>();

    bool x;
    auto l = lhs.getValue();
    auto r = rhs.getValue();
    switch (pred) {
    case CmpPredicate::LT:
      x = l.slt(r);
      break;
    case CmpPredicate::LTEQ:
      x = l.sle(r);
      break;
    case CmpPredicate::GT:
      x = l.sgt(r);
      break;
    case CmpPredicate::GTEQ:
      x = l.sge(r);
      break;
    case CmpPredicate::EQ:
      x = l == r;
      break;
    case CmpPredicate::NEQ:
      x = l != r;
      break;
    }

    auto result = mlir::BoolAttr::get(getContext(), x);
    return result;
  }

  return {};
}

struct SimplifyAddConstAddConst : public mlir::OpRewritePattern<AddIOp> {
  SimplifyAddConstAddConst(mlir::MLIRContext *context)
      : OpRewritePattern<AddIOp>(context, /*benefit=*/1) {}

  mlir::LogicalResult
  matchAndRewrite(AddIOp op, mlir::PatternRewriter &rewriter) const override {
    auto lhs = op.lhs().getDefiningOp<AddIOp>();
    if (!lhs)
      return mlir::failure();

    auto lConst = lhs.rhs().getDefiningOp<ConstantOp>();
    auto rConst = op.rhs().getDefiningOp<ConstantOp>();
    if (!lConst || !rConst)
      return mlir::failure();

    auto sumConst = rewriter.create<ConstantOp>(
        mlir::FusedLoc::get(op->getContext(),
                            {lConst->getLoc(), rConst.getLoc()}, nullptr),
        lConst.getType(),
        mlir::IntegerAttr::get(
            lConst.getType(),
            lConst.value().cast<mlir::IntegerAttr>().getValue() +
                rConst.value().cast<mlir::IntegerAttr>().getValue()));
    rewriter.replaceOpWithNewOp<AddIOp>(op, lhs.result().getType(), lhs.lhs(),
                                        sumConst);
    return mlir::success();
  }
};

void AddIOp::getCanonicalizationPatterns(mlir::RewritePatternSet &patterns,
                                         mlir::MLIRContext *context) {
  patterns.add<SimplifyAddConstAddConst>(context);
}

mlir::LogicalResult ArrayRefOp::verify() {
  auto type = getType();
  // The value's type must match the return type.
  auto arrayType = array().getType();

  if (!arrayType.isa<ir::ArrayType>()) {
    return emitOpError() << "ArrayRef requires an array as input: " << arrayType;
  }

  if (arrayType.cast<ir::ArrayType>().getElementType() != type) {
    return emitOpError() << "ArrayRef return type is not the array element type: array=" << arrayType << ", result=" << type;
  }

  return mlir::success();
}


mlir::OpFoldResult
ArrayRefOp::fold(llvm::ArrayRef<mlir::Attribute> operands) {
  auto a = array().getDefiningOp<MakeArrayOp>();
  if (operands[1].isa<mlir::IntegerAttr>() && a) {
    auto idx = operands[1].cast<mlir::IntegerAttr>().getInt();
    if (idx < 0 || idx >= a->getNumOperands())
      return {};
    return a->getOperands()[idx];
  }
  return {};
}

mlir::LogicalResult MakeArrayOp::verify() {
  auto assignedResultType = result().getType();

  if (!assignedResultType.isa<ir::ArrayType>()) {
    return emitOpError() << "MakeArray expects an ArrayType as return type, found " << assignedResultType;
  }

  auto elemType = assignedResultType.cast<ir::ArrayType>().getElementType();
  for (auto elem : elems()) {
    if (elemType != elem.getType()) {
      return emitOpError() << "MakeArray with return element type " << elemType << " had element with type " << elem.getType();
    }
  }
  return mlir::success();
}

} // namespace ir
} // namespace hail
