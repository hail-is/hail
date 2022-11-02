#ifndef DIALECT_HAIL_HAILDIALECTBASE
#define DIALECT_HAIL_HAILDIALECTBASE

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

def Hail_Dialect : Dialect {
  let name = "hail";
  let summary = "Dialect for defining operations of the hail query system";
  let cppNamespace = "::hail::ir";
  let useDefaultTypePrinterParser = 1;
}

class Hail_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Hail_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

// Here is a simple definition of an "integer" type, with a width parameter.
def Hail_Int : Hail_Type<"Integer", "int"> {
  let summary = "Hail 32 bit integer type";
  let description = [{
    Hail 32 bit integer type
  }];

  /// Indicate that our type will add additional verification to the parameters.
  let genVerifyDecl = 0;
}

#endif // DIALECT_HAIL_HAILDIALECTBASE
