#ifndef DIALECT_CPS_IR_CPSBASE
#define DIALECT_CPS_IR_CPSBASE

include "mlir/IR/AttrTypeBase.td"

def CPS_Dialect : Dialect {
  let name = "cps";
  let summary = "Allows for blocks of code in continuation passing style";
  let cppNamespace = "::hail::ir";
  let useDefaultTypePrinterParser = 1;
}

class CPS_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<CPS_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def CPS_ContType : CPS_Type<"Continuation", "cont"> {
  let summary = "Continuation type";
  let parameters = (ins ArrayRefParameter<"mlir::Type">:$inputs);
  // The declarative format "`<` $inputs `>`" doesn't support the empty list of types
  let hasCustomAssemblyFormat = 1;
}

#endif // DIALECT_CPS_IR_CPSBASE