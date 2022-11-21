#ifndef DIALECT_OPTION_IR_OPTIONOPS
#define DIALECT_OPTION_IR_OPTIONOPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltInTypes.td"

include "Dialect/CPS/IR/CPSBase.td"


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
  let assemblyFormat = "`<` $valueTypes `>`";
}

class Option_Op<string mnemonic, list<Trait> traits = []> :
        Op<Option_Dialect, mnemonic, traits>;

// def Option_DestructOp : Option_Op<"destruct", [NoSideEffect]> {
//   let arguments = (ins Option_OptionType:$input);
//   let results = (outs Variadic<AnyType>);
//   let regions = (region SizedRegion<1>:$missingRegion, SizedRegion<1>:$presentRegion);

//   let assemblyFormat = "$input `:` functional-type($input, results) attr-dict-with-keyword $missingRegion $presentRegion";
//   // let hasCustomAssemblyFormat = 1;
//   // let hasVerifier = 1;
// }

def Option_DestructOp : Option_Op<"destruct", [
      NoSideEffect, Terminator,
      TypesMatchWith<"", "input", "missingCont", "::hail::ir::ContinuationType::get($_ctxt, {})">,
      TypesMatchWith<"", "input", "presentCont", "::hail::ir::ContinuationType::get($_ctxt, $_self.cast<::hail::ir::OptionType>().getValueTypes())">
    ]> {
  let arguments = (ins Option_OptionType:$input, CPS_EmptyContType:$missingCont, CPS_ContType:$presentCont);

  let assemblyFormat = "$input`(` $missingCont`,` $presentCont `)` `:` qualified(type($input)) attr-dict";
  // let hasCustomAssemblyFormat = 1;
  // let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def Option_ConstructOp : Option_Op<"construct", [NoSideEffect]> {
  let results = (outs Option_OptionType:$result);
  let regions = (region SizedRegion<1>:$bodyRegion);

  let assemblyFormat = "qualified(type($result)) attr-dict-with-keyword $bodyRegion";
  // let hasCustomAssemblyFormat = 1;
}

#endif // DIALECT_OPTION_IR_OPTIONOPS
