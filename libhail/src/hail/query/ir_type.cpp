#include <hail/tunion.hpp>
#include <hail/query/ir.hpp>
#include <hail/query/ir_type.hpp>

namespace hail {

const Type *
IRType::infer(Block *x) {
  if (x->get_function_parent()) {
    std::vector<const Type *> input_types, output_types;
    for (ssize_t i = x->get_function_parent()->parameter_types.size(); i >= 0; --i)
      input_types.push_back(x->get_function_parent()->parameter_types[i]);
    for (ssize_t i = x->get_children().size() - 1; i >= 0; --i)
      output_types.push_back(infer(x->get_child(i)));
    return tc.tblock(input_types, output_types);
  }

  IR *parent = x->get_parent();
  switch (parent->tag) {
  default:
    abort();
  }
}

const Type *
IRType::infer(Input *x) {
  return cast<TBlock>(infer(x->get_parent()))
    ->input_types[x->index];
}

const Type *
IRType::infer(Literal *x) {
  return x->value.vtype->type;
}

const Type *
IRType::infer(NA *x) {
  return x->type;
}

const Type *
IRType::infer(IsNA *x) {
  return tc.tbool;
}

const Type *
IRType::infer(IR *x) {
  auto p = ir_type.insert({x, nullptr});
  if (!p.second)
    return p.first->second;
  
  const Type *t = x->dispatch([this](auto x) {
				return infer(x);
			      });
  p.first->second = t;
  return t;
}

IRType::IRType(TypeContext &tc, Function *f)
  : tc(tc) {
  infer(f->get_body());
}

}
