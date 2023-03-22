#include "hail/Conversion/LowerSandbox/LowerSandbox.h"

#include "../PassDetail.h"

#include "hail/Dialect/Sandbox/IR/Sandbox.h"
#include "hail/Support/MLIR.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"

namespace hail::ir {

struct LowerSandboxPass : public LowerSandboxBase<LowerSandboxPass> {
  void runOnOperation() override;
};

namespace {
struct AddIOpConversion : public OpConversionPattern<AddIOp> {
  AddIOpConversion(MLIRContext *context) : OpConversionPattern<AddIOp>(context, /*benefit=*/1) {}

  auto matchAndRewrite(AddIOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const
      -> LogicalResult override {
    rewriter.replaceOpWithNewOp<mlir::arith::AddIOp>(op, adaptor.lhs(), adaptor.rhs());
    return success();
  }
};

struct ConstantOpConversion : public OpConversionPattern<ConstantOp> {
  ConstantOpConversion(MLIRContext *context)
      : OpConversionPattern<ConstantOp>(context, /*benefit=*/1) {}

  auto matchAndRewrite(ConstantOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const
      -> LogicalResult override {
    auto value = adaptor.valueAttr().cast<IntegerAttr>().getValue();
    Attribute newAttr;
    Type newType;
    if (op.output().getType().isa<BooleanType>()) {
      newType = rewriter.getI1Type();
      newAttr = rewriter.getBoolAttr(value == 0);
    } else {
      newType = rewriter.getI32Type();
      newAttr = rewriter.getIntegerAttr(newType, value);
    }
    rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, newAttr, newType);
    return success();
  }
};

struct ComparisonOpConversion : public OpConversionPattern<ComparisonOp> {
  ComparisonOpConversion(MLIRContext *context)
      : OpConversionPattern<ComparisonOp>(context, /*benefit=*/1) {}

  auto matchAndRewrite(ComparisonOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const -> LogicalResult override {
    mlir::arith::CmpIPredicate pred{};

    switch (adaptor.predicate()) {
    case CmpPredicate::LT:
      pred = mlir::arith::CmpIPredicate::slt;
      break;
    case CmpPredicate::LTEQ:
      pred = mlir::arith::CmpIPredicate::sle;
      break;
    case CmpPredicate::GT:
      pred = mlir::arith::CmpIPredicate::sgt;
      break;
    case CmpPredicate::GTEQ:
      pred = mlir::arith::CmpIPredicate::sge;
      break;
    case CmpPredicate::EQ:
      pred = mlir::arith::CmpIPredicate::eq;
      break;
    case CmpPredicate::NEQ:
      pred = mlir::arith::CmpIPredicate::ne;
      break;
    }

    rewriter.replaceOpWithNewOp<mlir::arith::CmpIOp>(op, pred, adaptor.lhs(), adaptor.rhs());
    return success();
  }
};

auto getLoweredType(Builder &b, Type t) -> Type {
  if (t.isa<IntType>())
    return b.getI32Type();

  if (t.isa<BooleanType>())
    return b.getI1Type();

  if (t.isa<ArrayType>()) {
    auto loweredElem = getLoweredType(b, t.cast<ArrayType>().getElementType());
    SmallVector<int64_t, 1> const v = {-1};

    return RankedTensorType::get(v, loweredElem);
  }

  return {};
}

struct MakeArrayOpConversion : public OpConversionPattern<MakeArrayOp> {
  MakeArrayOpConversion(MLIRContext *context)
      : OpConversionPattern<MakeArrayOp>(context, /*benefit=*/1) {}

  auto matchAndRewrite(MakeArrayOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const
      -> LogicalResult override {

    auto elems = adaptor.elems();
    SmallVector<int64_t, 1> const v = {op->getNumOperands()};
    Type const loweredElem =
        op.getNumOperands() > 0
            ? adaptor.elems()[0].getType()
            : getLoweredType(rewriter, op.result().getType().cast<ArrayType>().getElementType());
    auto tensorType = RankedTensorType::get(v, loweredElem);
    rewriter.replaceOpWithNewOp<mlir::tensor::FromElementsOp>(op, tensorType, elems);

    return success();
  }
};

struct ArrayRefOpConversion : public OpConversionPattern<ir::ArrayRefOp> {
  ArrayRefOpConversion(MLIRContext *context)
      : OpConversionPattern<ArrayRefOp>(context, /*benefit=*/1) {}

  auto matchAndRewrite(ArrayRefOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const
      -> LogicalResult override {

    auto a = adaptor.array();
    auto idx = adaptor.index();
    rewriter.replaceOpWithNewOp<mlir::tensor::ExtractOp>(op, a, idx);

    return success();
  }
};

struct PrintOpConversion : public OpConversionPattern<PrintOp> {
  PrintOpConversion(MLIRContext *context) : OpConversionPattern<PrintOp>(context, /*benefit=*/1) {}

  auto matchAndRewrite(PrintOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const
      -> LogicalResult override {
    rewriter.replaceOpWithNewOp<PrintOp>(op, adaptor.value());
    return success();
  }
};

struct UnrealizedCastConversion : public OpConversionPattern<mlir::UnrealizedConversionCastOp> {
  UnrealizedCastConversion(MLIRContext *context)
      : OpConversionPattern<mlir::UnrealizedConversionCastOp>(context,
                                                              /*benefit=*/1) {}

  auto matchAndRewrite(mlir::UnrealizedConversionCastOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const -> LogicalResult override {
    rewriter.replaceOp(op, adaptor.getInputs());
    return success();
  }
};

} // end namespace

void LowerSandboxPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateLowerSandboxConversionPatterns(patterns);

  // Configure conversion to lower out SCF operations.
  ConversionTarget target(getContext());
  target.addIllegalDialect<SandboxDialect>();
  target.addDynamicallyLegalOp<PrintOp, mlir::UnrealizedConversionCastOp>([](Operation *op) {
    auto cond = [](Type type) {
      return type.isa<IntType>() || type.isa<BooleanType>() || type.isa<ArrayType>();
    };
    return llvm::none_of(op->getOperandTypes(), cond) && llvm::none_of(op->getResultTypes(), cond);
  });
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

void populateLowerSandboxConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<ConstantOpConversion, AddIOpConversion, ComparisonOpConversion, PrintOpConversion,
               UnrealizedCastConversion, MakeArrayOpConversion, ArrayRefOpConversion>(
      patterns.getContext());
}

auto createLowerSandboxPass() -> std::unique_ptr<Pass> {
  return std::make_unique<LowerSandboxPass>();
}

} // namespace hail::ir
