#ifndef DIALECT_HAIL_HAILOPS
#define DIALECT_HAIL_HAILOPS

include "Dialect/Hail/IR/HailDialectBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


class Hail_Op<string mnemonic, list<Trait> traits = []> :
        Op<Hail_Dialect, mnemonic, traits>;

def I32Op : Hail_Op<"i32", [ConstantLike, NoSideEffect]> {
  let arguments = (ins SI32Attr:$value);
  let results = (outs Hail_Int:$output);
  let assemblyFormat = "$value attr-dict `:` type($output)";

  let hasFolder = true;
}

def I32Plus : Hail_Op<"i32_plus", [NoSideEffect, Commutative]> {
  let arguments = (ins Hail_Int:$left, Hail_Int:$right);
  let results = (outs Hail_Int:$output);
  let assemblyFormat = "$left $right attr-dict";

  let hasFolder = true;
  let hasCanonicalizer = true;
}

def Print : Hail_Op<"print"> {
  let arguments = (ins Hail_Int:$value);
  let assemblyFormat = "$value attr-dict";
}

#endif // DIALECT_HAIL_HAILOPS
