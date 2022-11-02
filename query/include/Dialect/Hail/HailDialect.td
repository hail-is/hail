#ifndef DIALECT_HAIL_HAILDIALECT
#define DIALECT_HAIL_HAILDIALECT

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

class Hail_Op<string mnemonic, list<Trait> traits = []> :
        Op<Hail_Dialect, mnemonic, traits>;

def I32Op : Hail_Op<"i32"> {
  let arguments = (ins SI32Attr:$value);
  let results = (outs Hail_Int:$output);
  let assemblyFormat = "$value attr-dict `:` type($output)";
}


#endif // DIALECT_HAIL_HAILDIALECT
