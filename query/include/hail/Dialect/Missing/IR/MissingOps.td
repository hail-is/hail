#ifndef DIALECT_MISSING_MISSINGOPS
#define DIALECT_MISSING_MISSINGOPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def Missing_Dialect : Dialect {
  let name = "missing";
  let summary = "Allows inserting a new 'missing' value into all types";
  let cppNamespace = "::hail::ir";
}

class Missing_Op<string mnemonic, list<Trait> traits = []> :
        Op<Missing_Dialect, mnemonic, traits>;

def Missing_MissingOp : Missing_Op<"missing", [NoSideEffect]> {
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";

  // let builders = [
  //   OpBuilder<(ins "::mlir::Attribute":$value, "::mlir::Type":$type),
  //   [{ build($_builder, $_state, type, value); }]>,
  // ];
}

def Missing_IsMissingOp : Missing_Op<"is_missing", [NoSideEffect]> {
  let arguments = (ins AnyType:$operand);
  let results = (outs I1);
  let assemblyFormat = "$operand attr-dict `:` type($operand)";
}

#endif // DIALECT_MISSING_MISSINGOPS
