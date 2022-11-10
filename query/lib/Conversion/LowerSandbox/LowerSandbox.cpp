#include "Conversion/LowerSandbox/LowerSandbox.h"
#include "../PassDetail.h"
#include "Dialect/Sandbox/IR/Sandbox.h"

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
struct AddIOpConversion : public mlir::OpConversionPattern<ir::AddIOp> {
  AddIOpConversion(mlir::MLIRContext *context)
      : OpConversionPattern<ir::AddIOp>(context, /*benefit=*/1) {}

  mlir::LogicalResult
  matchAndRewrite(ir::AddIOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::arith::AddIOp>(op, adaptor.lhs(),
                                                     adaptor.rhs());
    return mlir::success();
  }
};

struct ConstantOpConversion : public mlir::OpConversionPattern<ir::ConstantOp> {
  ConstantOpConversion(mlir::MLIRContext *context)
      : OpConversionPattern<ir::ConstantOp>(context, /*benefit=*/1) {}

  mlir::LogicalResult
  matchAndRewrite(ir::ConstantOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto value = adaptor.valueAttr().cast<mlir::IntegerAttr>().getValue();
    mlir::Attribute newAttr;
    mlir::Type newType;
    if (op.output().getType().isa<ir::BooleanType>()) {
      newType = rewriter.getI1Type();
      newAttr = rewriter.getBoolAttr(value == 0);
    } else {
      newType = rewriter.getI32Type();
      newAttr = rewriter.getIntegerAttr(newType, value);
    }
    rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, newAttr, newType);
    return mlir::success();
  }
};

struct ComparisonOpConversion
    : public mlir::OpConversionPattern<ir::ComparisonOp> {
  ComparisonOpConversion(mlir::MLIRContext *context)
      : OpConversionPattern<ir::ComparisonOp>(context, /*benefit=*/1) {}

  mlir::LogicalResult
  matchAndRewrite(ir::ComparisonOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::arith::CmpIPredicate pred;

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

    rewriter.replaceOpWithNewOp<mlir::arith::CmpIOp>(op, pred, adaptor.lhs(),
                                                     adaptor.rhs());
    return mlir::success();
  }
};

mlir::Type getLoweredType(mlir::Builder &b, mlir::Type t) {
  if (t.isa<ir::IntType>())
    return b.getI32Type();

  if (t.isa<ir::BooleanType>())
    return b.getI1Type();

  if (t.isa<ir::ArrayType>()) {
    auto loweredElem =
        getLoweredType(b, t.cast<ir::ArrayType>().getElementType());
    llvm::SmallVector<int64_t, 1> v = {-1};

    return mlir::RankedTensorType::get(v, loweredElem);
  }

  return nullptr;
}

struct MakeArrayOpConversion
    : public mlir::OpConversionPattern<ir::MakeArrayOp> {
  MakeArrayOpConversion(mlir::MLIRContext *context)
      : OpConversionPattern<ir::MakeArrayOp>(context, /*benefit=*/1) {}

  mlir::LogicalResult
  matchAndRewrite(ir::MakeArrayOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    auto elems = adaptor.elems();
    llvm::SmallVector<int64_t, 1> v = {op->getNumOperands()};
    mlir::Type loweredElem = op.getNumOperands() > 0 ? adaptor.elems()[0].getType() : getLoweredType(rewriter, op.result().getType().cast<ir::ArrayType>().getElementType());
    auto tensorType = mlir::RankedTensorType::get(v, loweredElem);
    rewriter.replaceOpWithNewOp<mlir::tensor::FromElementsOp>(op, tensorType,
                                                              elems);

    return mlir::success();
  }
};

struct ArrayRefOpConversion
    : public mlir::OpConversionPattern<ir::ArrayRefOp> {
  ArrayRefOpConversion(mlir::MLIRContext *context)
      : OpConversionPattern<ir::ArrayRefOp>(context, /*benefit=*/1) {}

  mlir::LogicalResult
  matchAndRewrite(ir::ArrayRefOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    auto a = adaptor.array();
    auto idx = adaptor.index();
    rewriter.replaceOpWithNewOp<mlir::tensor::ExtractOp>(op, a, idx);

    return mlir::success();
  }
};


struct PrintOpConversion : public mlir::OpConversionPattern<ir::PrintOp> {
  PrintOpConversion(mlir::MLIRContext *context)
      : OpConversionPattern<ir::PrintOp>(context, /*benefit=*/1) {}

  mlir::LogicalResult
  matchAndRewrite(ir::PrintOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ir::PrintOp>(op, adaptor.value());
    return mlir::success();
  }
};

struct UnrealizedCastConversion
    : public mlir::OpConversionPattern<mlir::UnrealizedConversionCastOp> {
  UnrealizedCastConversion(mlir::MLIRContext *context)
      : OpConversionPattern<mlir::UnrealizedConversionCastOp>(context,
                                                              /*benefit=*/1) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getInputs());
    return mlir::success();
  }
};

} // end namespace

void LowerSandboxPass::runOnOperation() {
  mlir::RewritePatternSet patterns(&getContext());
  populateLowerSandboxConversionPatterns(patterns);

  // Configure conversion to lower out SCF operations.
  mlir::ConversionTarget target(getContext());
  target.addIllegalDialect<ir::SandboxDialect>();
  target.addDynamicallyLegalOp<ir::PrintOp, mlir::UnrealizedConversionCastOp>(
      [](mlir::Operation *op) {
        auto cond = [](mlir::Type type) {
          return type.isa<ir::IntType>() || type.isa<ir::BooleanType>() || type.isa<ir::ArrayType>();
        };
        return llvm::none_of(op->getOperandTypes(), cond) &&
               llvm::none_of(op->getResultTypes(), cond);
      });
  target.markUnknownOpDynamicallyLegal([](mlir::Operation *) { return true; });
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

void populateLowerSandboxConversionPatterns(mlir::RewritePatternSet &patterns) {
  patterns.add<ConstantOpConversion, AddIOpConversion, ComparisonOpConversion,
               PrintOpConversion, UnrealizedCastConversion, MakeArrayOpConversion,
               ArrayRefOpConversion>(
      patterns.getContext());
}

std::unique_ptr<mlir::Pass> createLowerSandboxPass() {
  return std::make_unique<LowerSandboxPass>();
}

} // namespace hail::ir