#include "hail/Dialect/Sandbox/IR/Sandbox.h"

#include "hail/Support/MLIR.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

#include <cstddef>

#include "hail/Dialect/Sandbox/IR/SandboxOpsEnums.cpp.inc"

#define GET_OP_CLASSES

#include "hail/Dialect/Sandbox/IR/SandboxOps.cpp.inc"

namespace hail::ir {

auto ConstantOp::fold(ArrayRef<Attribute> operands) -> OpFoldResult { return valueAttr(); }

auto ConstantOp::verify() -> LogicalResult {
  auto type = getType();
  // The value's type must match the return type.
  auto valueType = value().getType();

  if (valueType.isa<IntegerType>() && type.isa<IntType>()) {
    return success();
  }

  if (valueType == IntegerType::get(getContext(), 1) && type.isa<BooleanType>()) {
    return success();
  }

  return emitOpError() << "bad constant: type=" << type << ", valueType=" << valueType;
}

// NOLINTNEXTLINE(*-member-functions-to-static)
auto AddIOp::fold(ArrayRef<Attribute> operands) -> OpFoldResult {
  assert(operands.size() == 2 && "binary op takes two operands");
  if (!operands[0] || !operands[1])
    return {};

  if (operands[0].isa<IntegerAttr>() && operands[1].isa<IntegerAttr>()) {
    auto lhs = operands[0].cast<IntegerAttr>();
    auto rhs = operands[1].cast<IntegerAttr>();

    auto result = IntegerAttr::get(lhs.getType(), lhs.getValue() + rhs.getValue());
    assert(result);
    return result;
  }

  return {};
}

auto ComparisonOp::fold(ArrayRef<Attribute> operands) -> OpFoldResult {
  assert(operands.size() == 2 && "comparison op takes two operands");
  if (!operands[0] || !operands[1])
    return {};

  if (operands[0].isa<IntegerAttr>() && operands[1].isa<IntegerAttr>()) {
    auto pred = predicate();
    auto lhs = operands[0].cast<IntegerAttr>();
    auto rhs = operands[1].cast<IntegerAttr>();

    bool x = false;
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

    auto result = BoolAttr::get(getContext(), x);
    return result;
  }

  return {};
}

struct SimplifyAddConstAddConst : public OpRewritePattern<AddIOp> {
  SimplifyAddConstAddConst(MLIRContext *context)
      : OpRewritePattern<AddIOp>(context, /*benefit=*/1) {}

  auto matchAndRewrite(AddIOp op, PatternRewriter &rewriter) const -> LogicalResult override {
    auto lhs = op.lhs().getDefiningOp<AddIOp>();
    if (!lhs)
      return failure();

    auto lConst = lhs.rhs().getDefiningOp<ConstantOp>();
    auto rConst = op.rhs().getDefiningOp<ConstantOp>();
    if (!lConst || !rConst)
      return failure();

    auto sumConst = rewriter.create<ConstantOp>(
        mlir::FusedLoc::get(op->getContext(), {lConst->getLoc(), rConst.getLoc()}, nullptr),
        lConst.getType(),
        IntegerAttr::get(lConst.getType(), lConst.value().cast<IntegerAttr>().getValue()
                                               + rConst.value().cast<IntegerAttr>().getValue()));
    rewriter.replaceOpWithNewOp<AddIOp>(op, lhs.result().getType(), lhs.lhs(), sumConst);
    return success();
  }
};

void AddIOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) {
  results.add<SimplifyAddConstAddConst>(context);
}

auto ArrayRefOp::verify() -> LogicalResult {
  auto type = getType();
  // The value's type must match the return type.
  auto arrayType = array().getType();

  if (!arrayType.isa<ArrayType>()) {
    return emitOpError() << "ArrayRef requires an array as input: " << arrayType;
  }

  if (arrayType.cast<ArrayType>().getElementType() != type) {
    return emitOpError() << "ArrayRef return type is not the array element type: array="
                         << arrayType << ", result=" << type;
  }

  return success();
}

auto ArrayRefOp::fold(ArrayRef<Attribute> operands) -> OpFoldResult {
  auto a = array().getDefiningOp<MakeArrayOp>();
  if (operands[1].isa<IntegerAttr>() && a) {
    auto idx = operands[1].cast<IntegerAttr>().getInt();
    if (idx < 0 || idx >= a->getNumOperands())
      return {};
    return a->getOperands()[idx];
  }
  return {};
}

auto MakeArrayOp::verify() -> LogicalResult {
  auto assignedResultType = result().getType();

  if (!assignedResultType.isa<ArrayType>()) {
    return emitOpError() << "MakeArray expects an ArrayType as return type, found "
                         << assignedResultType;
  }

  auto elemType = assignedResultType.cast<ArrayType>().getElementType();
  for (auto elem : elems()) {
    if (elemType != elem.getType()) {
      return emitOpError() << "MakeArray with return element type " << elemType
                           << " had element with type " << elem.getType();
    }
  }
  return success();
}

} // namespace hail::ir
