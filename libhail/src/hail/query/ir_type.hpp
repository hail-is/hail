#ifndef HAIL_IR_TYPE_HPP_INCLUDED
#define HAIL_IR_TYPE_HPP_INCLUDED 1

#include <unordered_map>

#include "hail/query/ir.hpp"

namespace hail {

class Type;
class TypeContext;
class Function;
class IR;

class IRType {
  friend class IRTypeVisitor;

  TypeContext &tc;

  std::unordered_map<IR *, const Type *> ir_type;

  const Type *infer(Block *x);
  const Type *infer(Input *x);
  const Type *infer(Literal *x);
  const Type *infer(NA *x);
  const Type *infer(IsNA *x);
  const Type *infer(MakeTuple *x);
  const Type *infer(GetTupleElement *x);
  const Type *infer(IR *x);

public:
  IRType(TypeContext &tc, Function *f);

  const Type *operator()(IR *x) { return ir_type.find(x)->second; }
};

}

#endif
