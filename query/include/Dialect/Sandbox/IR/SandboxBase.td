#ifndef DIALECT_SANDBOX_SANDBOXBASE
#define DIALECT_SANDBOX_SANDBOXBASE

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"


include "mlir/IR/EnumAttr.td"


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

class Sandbox_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Sandbox_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def SandboxType :
    DialectType<Sandbox_Dialect, CPred<"$_self.isa<Sandbox_Type>()">,
                "Sandbox type", "::hail::ir::SandboxType">;

def Sandbox_OptionalAttr : Sandbox_Attr<"Optional", "opt"> {
  let summary = "Sandbox attribute representing a possibly missing value";
  let parameters = (ins AttributeSelfTypeParameter<"">:$type, OptionalParameter<"mlir::Attribute">:$value);
  // let builders = [
  //   AttrBuilderWithInferredContext<(ins "Type":$type,
  //                                       "const Attribute &":$value), [{
  //     return $_get(type.getContext(), type, value);
  //   }]>
  // ];
  let assemblyFormat = "`<` $value `>`";
}

// Here is a simple definition of an "integer" type, with a width parameter.
def Sandbox_Int : Sandbox_Type<"Int", "int"> {
  let summary = "Sandbox 32 bit integer type";
  let description = [{
    Sandbox 32 bit integer type
  }];

  /// Indicate that our type will add additional verification to the parameters.
  let genVerifyDecl = 0;
}

def Sandbox_Bool : Sandbox_Type<"Boolean", "bool"> {
  let summary = "Sandbox 32 bit integer type";
  let description = [{
    Sandbox 32 bit integer type
  }];

  /// Indicate that our type will add additional verification to the parameters.
  let genVerifyDecl = 0;
}

def Sandbox_CmpPredicateAttr : I64EnumAttr<
    "CmpPredicate", "",
    [
      I64EnumAttrCase<"LT", 0, "lt">,
      I64EnumAttrCase<"LTEQ", 1, "lteq">,
      I64EnumAttrCase<"GT", 2, "gt">,
      I64EnumAttrCase<"GTEQ", 3, "gteq">,
      I64EnumAttrCase<"EQ", 4, "eq">,
      I64EnumAttrCase<"NEQ", 5, "eq">,
    ]> {
}

// def Sandbox_Array : Sandbox_Type<"Array", "array", [
//     DeclareTypeInterfaceMethods<SubElementTypeInterface>
//   ]> {
//   let summary = "Sandbox array";
//   let parameters = (ins "mlir::Type":$elementType);

//   // let builders = [
//   //   TypeBuilderWithInferredContext<(ins "Type":$elementType), [{
//   //     // Drop default memory space value and replace it with empty attribute.
//   //     Attribute nonDefaultMemorySpace = skipDefaultMemorySpace(memorySpace);
//   //     return $_get(elementType.getContext(), elementType, nonDefaultMemorySpace);
//   //   }]>,
//   //   /// [deprecated] `Attribute`-based form should be used instead.
//   //   TypeBuilderWithInferredContext<(ins "Type":$elementType,
//   //                                       "unsigned":$memorySpace), [{
//   //     // Convert deprecated integer-like memory space to Attribute.
//   //     Attribute memorySpaceAttr =
//   //         wrapIntegerMemorySpace(memorySpace, elementType.getContext());
//   //     return UnrankedMemRefType::get(elementType, memorySpaceAttr);
//   //   }]>
//   // ];
//   let extraClassDeclaration = [{
//     // using ShapedType::Trait<UnrankedMemRefType>::clone;
//     // using ShapedType::Trait<UnrankedMemRefType>::getElementTypeBitWidth;
//     // using ShapedType::Trait<UnrankedMemRefType>::getRank;
//     // using ShapedType::Trait<UnrankedMemRefType>::getNumElements;
//     // using ShapedType::Trait<UnrankedMemRefType>::isDynamicDim;
//     // using ShapedType::Trait<UnrankedMemRefType>::hasStaticShape;
//     // using ShapedType::Trait<UnrankedMemRefType>::getNumDynamicDims;
//     // using ShapedType::Trait<UnrankedMemRefType>::getDimSize;
//     // using ShapedType::Trait<UnrankedMemRefType>::getDynamicDimIndex;

//     // ArrayRef<int64_t> getShape() const { return llvm::None; }

//     // /// [deprecated] Returns the memory space in old raw integer representation.
//     // /// New `Attribute getMemorySpace()` method should be used instead.
//     // unsigned getMemorySpaceAsInt() const;
//   }];
//   let skipDefaultBuilders = 1;
//   let mnemonic = ?;
//   let genVerifyDecl = 1;
// }

def Sandbox_Array : Sandbox_Type<"Array", "array", [
    DeclareTypeInterfaceMethods<SubElementTypeInterface>
  ]> {
  let summary = "Sandbox array";
  let parameters = (ins "mlir::Type":$elementType);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}


#endif // DIALECT_SANDBOX_SANDBOXBASE
