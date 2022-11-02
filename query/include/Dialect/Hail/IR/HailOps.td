#ifndef DIALECT_HAIL_HAILOPS
#define DIALECT_HAIL_HAILOPS

include "Dialect/Hail/IR/HailDialectBase.td"

class Hail_Op<string mnemonic, list<Trait> traits = []> :
        Op<Hail_Dialect, mnemonic, traits>;

def I32Op : Hail_Op<"i32"> {
  let arguments = (ins SI32Attr:$value);
  let results = (outs Hail_Int:$output);
  let assemblyFormat = "$value attr-dict `:` type($output)";
}

#endif // DIALECT_HAIL_HAILOPS
