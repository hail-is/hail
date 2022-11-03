#ifndef DIALECT_HAIL_HAILDIALECT
#define DIALECT_HAIL_HAILDIALECT

include "mlir/IR/OpBase.td"

def Hail_Dialect : Dialect {
  let name = "hail";
  let summary = "Dialect for defining operations of the hail query system";
  let cppNamespace = "::hail::ir";
}

class Hail_Op<string mnemonic, list<Trait> traits = []> :
        Op<Hail_Dialect, mnemonic, traits>;

#endif // DIALECT_HAIL_HAILDIALECT
