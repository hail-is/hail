#ifndef HAIL_CONVERSION_PASSES
#define HAIL_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// AffineToStandard
//===----------------------------------------------------------------------===//

def LowerSandbox : Pass<"lower-sandbox"> {
  let summary = "Lower sandbox operations to arith";
  let description = [{
    Convert operations and types from the sandbox dialect into operations and
    types from the arithmetic dialect.
  }];
  let constructor = "hail::ir::createLowerSandboxPass()";
  let dependentDialects = [
    "mlir::arith::ArithmeticDialect"
  ];
}

def LowerToLLVM : Pass<"lower-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lower all remaining operations to llvm";
  let description = [{
    Convert operations and types from the SCF, Arithmetic, and Func dialects,
    plus the sandbox print operation, to the LLVM dialect.
  }];
  let constructor = "hail::ir::createLowerToLLVMPass()";
  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "mlir::scf::SCFDialect"
  ];
}

#endif // HAIL_CONVERSION_PASSES