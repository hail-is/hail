#include <hail/tunion.hpp>
#include <hail/query/ir.hpp>
#include <hail/query/ir_type.hpp>

namespace hail {

const Type *
IRType::infer(Block *x) {
  abort();
  // FIXMEa add BlockType
#if 0
  const BlockType *tc.make_block_type(x->inputs.size(), x->children.size());
  if (x->function_parent) {
    std::vector<const Type *> input_types, output_types;
    for (ssize_t i = x->function_parent->paramter_types.size(); i >= 0; --i)
      input_types.push_back(x->function_parent->paramter_types[i]);
    for (ssize_t i = x->children.size(); i >= 0; --i)
      output_types[i] = infer(x->children[i]);
    return tc.make_block_type(input_types, output_types);
  }

  IR *parent = x->parent;
  switch (parent->tag) {
  default:
    abort();
  }
#endif
}

const Type *
IRType::infer(Input *x) {
  abort();
#if 0
  cast<BlockType>(infer(x->parent))
    ->input_types[x->index];
#endif
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
