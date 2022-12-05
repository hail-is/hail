#ifndef DIALECT_SANDBOX_SANDBOXOPS
#define DIALECT_SANDBOX_SANDBOXOPS

include "hail/Dialect/Sandbox/IR/SandboxBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

class Sandbox_Op<string mnemonic, list<Trait> traits = []> :
        Op<Sandbox_Dialect, mnemonic, traits>;

def Sandbox_ConstantOp : Sandbox_Op<"constant", [ConstantLike, NoSideEffect]> {
  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyType:$output);
  let assemblyFormat = "`(`$value`)` attr-dict `:` type($output)";

  let builders = [
    OpBuilder<(ins "::mlir::Attribute":$value, "::mlir::Type":$type),
    [{ build($_builder, $_state, type, value); }]>,
  ];

  let hasFolder = true;
  let hasVerifier = true;
}


def Sandbox_AddIOp : Sandbox_Op<"addi", [NoSideEffect, Commutative]> {
  let arguments = (ins Sandbox_Int:$lhs, Sandbox_Int:$rhs);
  let results = (outs Sandbox_Int:$result);
  let assemblyFormat = "$lhs $rhs attr-dict";

  let hasFolder = true;
  let hasCanonicalizer = true;
}

def Sandbox_ComparisonOp : Sandbox_Op<"compare", [NoSideEffect]> {
  let arguments = (ins Sandbox_CmpPredicateAttr:$predicate, Sandbox_Int:$lhs, Sandbox_Int:$rhs);
  let results = (outs Sandbox_Bool:$output);
  let assemblyFormat = "$predicate `,` $lhs `,` $rhs attr-dict `:` type($output)";

    let extraClassDeclaration = [{
    static CmpPredicate getPredicateByName(llvm::StringRef name);
  }];

  let hasFolder = 1;
  let hasCanonicalizer = 0;
}

def Sandbox_MakeArrayOp : Sandbox_Op<"make_array", [NoSideEffect]> {
  let arguments = (ins Variadic<AnyType>:$elems);
  let results = (outs AnyType:$result);

  let assemblyFormat = "`(` $elems `)` `(` type($elems) `)` attr-dict `:` type($result)";

  let hasVerifier = 1;
  let hasFolder = 0;
  let hasCanonicalizer = 0;
}

def Sandbox_ArrayRefOp : Sandbox_Op<"array_ref", [NoSideEffect]> {
  let arguments = (ins AnyType:$array, Sandbox_Int:$index);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$array type($array) $index attr-dict `:` type($result)";
  let hasVerifier = 1;

  let hasFolder = 1;
  let hasCanonicalizer = 0;
}

// def Sandbox_ArrayMap : Sandbox_Op<"array_map", [NoSideEffect]> {

// }

def Sandbox_PrintOp : Sandbox_Op<"print"> {
  let arguments = (ins AnyType:$value);
  let assemblyFormat = "$value attr-dict `:` type($value)";
}

#endif // DIALECT_SANDBOX_SANDBOXOPS
