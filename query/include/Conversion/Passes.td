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
  let constructor = "hail::createLowerSandboxPass()";
  let dependentDialects = [
    "hail::ir::SandboxDialect"
  ];
}

#endif // HAIL_CONVERSION_PASSES