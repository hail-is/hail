#ifndef DIALECT_CPS_IR_CPSOPS
#define DIALECT_CPS_IR_CPSOPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltInTypes.td"
include "mlir/IR/OpBase.td"

include "Dialect/CPS/IR/CPSBase.td"

class CPS_Op<string mnemonic, list<Trait> traits = []> :
        Op<CPS_Dialect, mnemonic, traits>;


def CPS_CallCCOp : CPS_Op<"callcc", [RecursiveSideEffects]> {
  let summary = "Call with current continuation";
  let description = [{
  }];

  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
}

def CPS_DefContOp : CPS_Op<"cont", [NoSideEffect, SingleBlock]> {
  let summary = "Define a continuation";
  let description = [{
  }];

  let results = (outs CPS_ContType:$result);
  let regions = (region SizedRegion<1>:$bodyRegion);

  let hasCustomAssemblyFormat = 1;
}

def CPS_ApplyContOp : CPS_Op<"apply", [Terminator]> {
  let summary = "Apply a continuation";
  let description = [{
  }];

  let arguments = (ins CPS_ContType:$cont, Variadic<AnyType>:$args);
  let hasCustomAssemblyFormat = 1;
}

#endif // DIALECT_CPS_IR_CPSOPS
