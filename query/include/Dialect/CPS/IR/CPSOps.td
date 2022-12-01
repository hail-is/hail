#ifndef DIALECT_CPS_IR_CPSOPS
#define DIALECT_CPS_IR_CPSOPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"

include "Dialect/CPS/IR/CPSBase.td"

class CPS_Op<string mnemonic, list<Trait> traits = []> :
        Op<CPS_Dialect, mnemonic, traits>;


def CPS_CallCCOp : CPS_Op<"callcc", [RecursiveSideEffects, SingleBlock]> {
  let summary = "Call with current continuation";
  let description = [{
  }];

  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
  let hasCanonicalizer = 1;

  let skipDefaultBuilders = 1;
  let builders = [ OpBuilder<(ins "mlir::TypeRange":$resultTypes)> ];
}

def CPS_DefContOp : CPS_Op<"cont", [NoSideEffect, SingleBlock]> {
  let summary = "Define a continuation";
  let description = [{
  }];

  let results = (outs CPS_ContType:$result);
  let regions = (region SizedRegion<1>:$bodyRegion);

  let hasCustomAssemblyFormat = 1;

  let skipDefaultBuilders = 1;
  let builders = [ OpBuilder<(ins CArg<"mlir::TypeRange", "mlir::TypeRange()">:$argTypes)> ];
}

def CPS_ApplyContOp : CPS_Op<"apply", [
    Terminator, NoSideEffect,
    TypesMatchWith<"", "args", "cont", "::hail::ir::ContinuationType::get($_ctxt, $_self)">
  ]> {
  let summary = "Apply a continuation";
  let description = [{
  }];

  let arguments = (ins CPS_ContType:$cont, Variadic<AnyType>:$args);
  let assemblyFormat = "$cont (`(` $args^ `)` `:` type($args))? attr-dict";

  let hasCanonicalizer = 1;
}

def CPS_IfOp : CPS_Op<"if", [
    Terminator, NoSideEffect, AttrSizedOperandSegments,
    TypesMatchWith<"", "trueArgs", "trueCont", "::hail::ir::ContinuationType::get($_ctxt, $_self)">,
    TypesMatchWith<"", "falseArgs", "falseCont", "::hail::ir::ContinuationType::get($_ctxt, $_self)">
  ]> {
  let summary = "if-then-else operation";
  let description = [{
  }];
  let arguments = (ins I1:$condition);
  let arguments = (ins I1:$condition,
                       CPS_ContType:$trueCont, Variadic<AnyType>:$trueArgs,
                       CPS_ContType:$falseCont, Variadic<AnyType>:$falseArgs);

  let assemblyFormat = [{
    $condition `,`
    $trueCont (`(` $trueArgs^ `:` type($trueArgs) `)`)? `,`
    $falseCont (`(` $falseArgs^ `:` type($falseArgs) `)`)?
    attr-dict
  }];
}

// FIXME: find the right dialect for this to live
def CPS_UndefinedOp : CPS_Op<"undefined", [NoSideEffect]> {
  let summary = "undefined";
  let description = [{
  }];

  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];
}

#endif // DIALECT_CPS_IR_CPSOPS
