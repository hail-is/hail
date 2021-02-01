#include <hail/tunion.hpp>
#include <hail/query/ir_type.hpp>

IRType::IRType(TypeContext &tc) {}

class IRTypeVisitor {
  IRType &ir_type;

  IRTypeVisitor(IRType &ir_type) : ir_type(ir_type) {}

  const Type *block(Block *x) {
    const BlockType *ir_type.tc.make_block_type(x->inputs.size(), x->children.size());
    if (x->function_parent) {
      std::vector<const Type *> input_types, output_types;
      for (ssize_t i = x->function_parent->paramter_types.size(); i >= 0; --i)
	input_types.push_back(x->function_parent->paramter_types[i]);
      for (ssize_t i = x->children.size(); i >= 0; --i)
	output_types[i] = visit(x->children[i]);
      return ir_type.tc.make_block_type(input_types, output_types);
    }

    IR *parent = x->parent;
    switch (parent->tag) {
    default:
      abort();
    }
  }

  const Type *visit(Input *x) {
    cast<BlockType>(ir_type(x->parent))
      ->input_types[x->index];
  }

  const Type *visit(Literal *x) {
    return x->value.get_type();
  }

  const Type *visit(NA *x) {
    return x->type;
  }

  const Type *visit(IsNA *x) {
    return ir_type.tc.tbool;
  }

  const Type *visit(IR *x) {
    const Type *t = x->dispatch(*this);
    ir_type.ir_type.insert({x, t});
    return t;
  }
};

IRType::IRType(TypeContext &tc, Function *f) {
  IRTypeVisitor().visit(f->get_body());
}
