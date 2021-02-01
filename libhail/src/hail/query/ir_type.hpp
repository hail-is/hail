#ifndef HAIL_IR_TYPE_HPP_INCLUDED
#define HAIL_IR_TYPE_HPP_INCLUDED 1

#include <unordered_map>

namespace hail {

class Type;
class TypeContext;
class IR;

class IRType {
  TypeContext &tc;

  std::unordered_map<IR *, const Type *> ir_type;
public:
  IRType(TypeContext &tc, Function *f);

  const Type *operator()(IR *x) { return ir_type.find(x)->second; }
};

}

#endif
