#ifndef DIALECT_OPTION_IR_OPTIONOPS
#define DIALECT_OPTION_IR_OPTIONOPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"

include "hail/Dialect/CPS/IR/CPSBase.td"


def Option_Dialect : Dialect {
  let name = "option";
  let summary = "Provides the Option type";
  let cppNamespace = "::hail::ir";
  let useDefaultTypePrinterParser = 1;
  let dependentDialects = ["CPSDialect"];
}

class Option_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Option_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Option_OptionType : Option_Type<"Option", "option"> {
  let summary = "Option type";
  let parameters = (ins ArrayRefParameter<"mlir::Type">:$valueTypes);
  // The declarative format "`<` $inputs `>`" doesn't support the empty list of types
  let hasCustomAssemblyFormat = 1;

  let builders = [
    TypeBuilder<(ins CArg<"mlir::TypeRange", "mlir::TypeRange()">:$inputTypes), [{
      return $_get($_ctxt, inputTypes);
    }]>
  ];
  let skipDefaultBuilders = 1;
  let genStorageClass = 0;
}

class Option_Op<string mnemonic, list<Trait> traits = []> :
        Op<Option_Dialect, mnemonic, traits>;

def Option_DestructOp : Option_Op<"destruct", [ NoSideEffect, Terminator ]> {
  let arguments = (ins Variadic<Option_OptionType>:$inputs, CPS_EmptyContType:$missingCont, CPS_ContType:$presentCont);

  let hasCustomAssemblyFormat = 1;
  let hasCanonicalizer = 1;
}

def Option_ConstructOp : Option_Op<"construct", [NoSideEffect, SingleBlock]> {
  let results = (outs Option_OptionType:$result);
  let regions = (region SizedRegion<1>:$bodyRegion);

  let assemblyFormat = "qualified(type($result)) attr-dict-with-keyword $bodyRegion";

  let skipDefaultBuilders = 1;
  let builders = [ OpBuilder<(ins CArg<"mlir::TypeRange", "mlir::TypeRange()">:$valueTypes)> ];

  let extraClassDeclaration = [{
    mlir::Value getMissingCont() { return getBody()->getArgument(0); }
    mlir::Value getPresentCont() { return getBody()->getArgument(0); }
  }];
}

def Option_MapOp : Option_Op<"map", [NoSideEffect, SingleBlock]> {
  let arguments = (ins Variadic<Option_OptionType>:$inputs);
  let results = (outs Option_OptionType:$result);
  let regions = (region SizedRegion<1>:$bodyRegion);

  let assemblyFormat = "`(` $inputs `)` `:` functional-type($inputs, $result) attr-dict-with-keyword $bodyRegion";

  let skipDefaultBuilders = 1;
  let builders = [
     OpBuilder<(ins CArg<"mlir::TypeRange", "mlir::TypeRange()">:$resultValueTypes,
                    CArg<"mlir::ValueRange", "mlir::ValueRange()">:$inputs)> ];
}

def Option_YieldOp : Option_Op<"yield", [NoSideEffect, Terminator]> {
  let arguments = (ins Variadic<AnyType>:$inputs);
  let assemblyFormat = "$inputs `:` type($inputs) attr-dict";
}

#endif // DIALECT_OPTION_IR_OPTIONOPS
