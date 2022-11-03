#ifndef DIALECT_SANDBOX_SANDBOXBASE
#define DIALECT_SANDBOX_SANDBOXBASE

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

def Sandbox_Dialect : Dialect {
  let name = "sb";
  let summary = "Dialect for experimenting with MLIR";
  let cppNamespace = "::hail::ir";
  let useDefaultTypePrinterParser = 1;
  let hasConstantMaterializer = 1;
}

class Sandbox_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Sandbox_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

// Here is a simple definition of an "integer" type, with a width parameter.
def Sandbox_Int : Sandbox_Type<"Int", "int"> {
  let summary = "Hail 32 bit integer type";
  let description = [{
    Hail 32 bit integer type
  }];

  /// Indicate that our type will add additional verification to the parameters.
  let genVerifyDecl = 0;
}

#endif // DIALECT_SANDBOX_SANDBOXBASE
