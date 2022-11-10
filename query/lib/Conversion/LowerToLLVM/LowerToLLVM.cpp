#include "Conversion/LowerToLLVM/LowerToLLVM.h"
#include "../PassDetail.h"
#include "Dialect/Sandbox/IR/Sandbox.h"

#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

namespace hail::ir {

struct LowerToLLVMPass : public LowerToLLVMBase<LowerToLLVMPass> {
  void runOnOperation() override;
};

namespace {

class PrintOpLowering : public mlir::OpConversionPattern<PrintOp> {
public:
  explicit PrintOpLowering(mlir::MLIRContext *context)
      : OpConversionPattern<ir::PrintOp>(context, /*benefit=*/1) {}

  mlir::LogicalResult
  matchAndRewrite(PrintOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    if (!(op.getOperand().getType() == rewriter.getI32Type() || op.getOperand().getType() == rewriter.getI1Type()))
      return mlir::failure();

    auto loc = op->getLoc();

    mlir::ModuleOp parentModule = op->getParentOfType<mlir::ModuleOp>();

    // Get a symbol reference to the printf function, inserting it if necessary.
    auto printfRef = getOrInsertPrintf(rewriter, parentModule);
    mlir::Value formatSpecifierCst = getOrCreateGlobalString(
        loc, rewriter, "frmt_spec", llvm::StringRef("%i \0", 4), parentModule);
    mlir::Value newLineCst = getOrCreateGlobalString(
        loc, rewriter, "nl", llvm::StringRef("\n\0", 2), parentModule);

    rewriter.create<mlir::func::CallOp>(
        loc, printfRef, rewriter.getIntegerType(32),
        llvm::ArrayRef<mlir::Value>({formatSpecifierCst, op.getOperand()}));
    rewriter.create<mlir::func::CallOp>(
        loc, printfRef, rewriter.getIntegerType(32), newLineCst);

    // Notify the rewriter that this operation has been removed.
    rewriter.eraseOp(op);
    return mlir::success();
  }

private:
  /// Return a symbol reference to the printf function, inserting it into the
  /// module if necessary.
  static mlir::FlatSymbolRefAttr
  getOrInsertPrintf(mlir::PatternRewriter &rewriter, mlir::ModuleOp module) {
    auto *context = module.getContext();
    if (module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("printf"))
      return mlir::SymbolRefAttr::get(context, "printf");

    // Create a function declaration for printf, the signature is:
    //   * `i32 (i8*, ...)`
    auto llvmI32Ty = mlir::IntegerType::get(context, 32);
    auto llvmI8PtrTy =
        mlir::LLVM::LLVMPointerType::get(mlir::IntegerType::get(context, 8));
    auto llvmFnType = mlir::LLVM::LLVMFunctionType::get(llvmI32Ty, llvmI8PtrTy,
                                                        /*isVarArg=*/true);

    // Insert the printf function into the body of the parent module.
    mlir::PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<mlir::LLVM::LLVMFuncOp>(module.getLoc(), "printf",
                                            llvmFnType);
    return mlir::SymbolRefAttr::get(context, "printf");
  }

  /// Return a value representing an access into a global string with the given
  /// name, creating the string if necessary.
  static mlir::Value getOrCreateGlobalString(mlir::Location loc,
                                             mlir::OpBuilder &builder,
                                             llvm::StringRef name,
                                             llvm::StringRef value,
                                             mlir::ModuleOp module) {
    // Create the global at the entry of the module.
    mlir::LLVM::GlobalOp global;
    if (!(global = module.lookupSymbol<mlir::LLVM::GlobalOp>(name))) {
      mlir::OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto type = mlir::LLVM::LLVMArrayType::get(
          mlir::IntegerType::get(builder.getContext(), 8), value.size());
      global = builder.create<mlir::LLVM::GlobalOp>(
          loc, type, /*isConstant=*/true, mlir::LLVM::Linkage::Internal, name,
          builder.getStringAttr(value),
          /*alignment=*/0);
    }

    // Get the pointer to the first character in the global string.
    mlir::Value globalPtr =
        builder.create<mlir::LLVM::AddressOfOp>(loc, global);
    mlir::Value cst0 = builder.create<mlir::LLVM::ConstantOp>(
        loc, mlir::IntegerType::get(builder.getContext(), 64),
        builder.getIntegerAttr(builder.getIndexType(), 0));
    return builder.create<mlir::LLVM::GEPOp>(
        loc,
        mlir::LLVM::LLVMPointerType::get(
            mlir::IntegerType::get(builder.getContext(), 8)),
        globalPtr, llvm::ArrayRef<mlir::Value>({cst0, cst0}));
  }
};

} // end namespace

void populateLowerLLVMConversionPatterns(mlir::LLVMTypeConverter &typeConverter,
                                         mlir::RewritePatternSet &patterns) {
  // populateSCFToControlFlowConversionPatterns(patterns);
  mlir::arith::populateArithmeticToLLVMConversionPatterns(typeConverter,
                                                          patterns);
  populateMemRefToLLVMConversionPatterns(typeConverter, patterns);
  mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                        patterns);
  populateFuncToLLVMConversionPatterns(typeConverter, patterns);

  // The only remaining operation to lower from the `toy` dialect, is the
  // PrintOp.
  patterns.add<PrintOpLowering>(patterns.getContext());
}

void LowerToLLVMPass::runOnOperation() {
  mlir::LLVMConversionTarget target(getContext());
  target.addLegalOp<mlir::ModuleOp>();

  mlir::RewritePatternSet patterns(&getContext());
  mlir::LLVMTypeConverter typeConverter(patterns.getContext());
  populateLowerLLVMConversionPatterns(typeConverter, patterns);

  // We want to completely lower to LLVM, so we use a `FullConversion`. This
  // ensures that only legal operations will remain after the conversion.
  auto module = getOperation();
  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> createLowerToLLVMPass() {
  return std::make_unique<LowerToLLVMPass>();
}

} // namespace hail::ir