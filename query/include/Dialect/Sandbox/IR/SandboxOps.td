#ifndef DIALECT_SANDBOX_SANDBOXOPS
#define DIALECT_SANDBOX_SANDBOXOPS

include "Dialect/Sandbox/IR/SandboxBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


class Sandbox_Op<string mnemonic, list<Trait> traits = []> :
        Op<Sandbox_Dialect, mnemonic, traits>;

def Sandbox_ConstantOp : Sandbox_Op<"constant", [ConstantLike, NoSideEffect]> {
  let arguments = (ins SI32Attr:$value);
  let results = (outs Sandbox_Int:$output);
  let assemblyFormat = "$value attr-dict `:` type($output)";

  let hasFolder = true;
}

def Sandbox_AddIOp : Sandbox_Op<"addi", [NoSideEffect, Commutative]> {
  let arguments = (ins Sandbox_Int:$lhs, Sandbox_Int:$rhs);
  let results = (outs Sandbox_Int:$result);
  let assemblyFormat = "$lhs $rhs attr-dict";

  let hasFolder = true;
  let hasCanonicalizer = true;
}

def BoolOp : Sandbox_Op<"bool", [ConstantLike, NoSideEffect]> {
  let arguments = (ins BoolAttr:$value);
  let results = (outs Sandbox_Bool:$output);
  let assemblyFormat = "$value attr-dict `:` type($output)";
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

def Sandbox_PrintOp : Sandbox_Op<"print"> {
  let arguments = (ins Sandbox_Int:$value);
  let assemblyFormat = "$value attr-dict";
}

#endif // DIALECT_SANDBOX_SANDBOXOPS
