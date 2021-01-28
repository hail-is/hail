#include <exception>
#include <hail/ir.hpp>
#include <hail/format.hpp>

namespace hail {

IRContext::IRContext(HeapAllocator &heap)
  : arena(heap) {}
    
IRContext::~IRContext() {}

Module *
IRContext::make_module() {
  return arena.make<Module>(IRContextToken());
}

Function *
IRContext::make_function(Module *module,
			 std::string name,
			 std::vector<const Type *> parameter_types,
			 const Type *return_type) {
  auto f = arena.make<Function>(IRContextToken(),
				module,
				std::move(name),
				std::move(parameter_types),
				return_type);
  f->set_body(arena.make<Block>(IRContextToken(), f, nullptr, 1, parameter_types.size()));
  return f;
}

Module::Module(IRContextToken) {}

Module::~Module() {}

void
Module::add_function(Function *f) {
  f->remove();
  f->module = this;
  assert(!functions.contains(f->name));
  functions.insert({f->name, f});
}

Function::Function(IRContextToken,
		   Module *module,
		   std::string name,
		   std::vector<const Type *> parameter_types,
		   const Type *return_type)
  : module(module),
    name(std::move(name)),
    parameter_types(std::move(parameter_types)),
    return_type(return_type) {
  module->add_function(this);
}

Function::~Function() {}

void
Function::remove() {
  if (!module)
    return;

  module->functions.erase(name);
  module = nullptr;
}

void
Function::set_body(Block *new_body) {
  if (body)
    body->remove();
  if (new_body)
    new_body->remove();
  body = new_body;
  body->function_parent = this;
}

IR::IR(Tag tag, IR *parent, size_t arity)
  : tag(tag), parent(parent), children(arity) {}

IR::IR(Tag tag, IR *parent, std::vector<IR *> children)
  : tag(tag), parent(parent), children(std::move(children)) {}

IR::~IR() {}

Block::Block(IRContextToken, Function *function_parent, IR *parent, size_t arity, size_t input_arity)
  : IR(self_tag, parent, arity),
    function_parent(function_parent),
    inputs(input_arity) {
}

void
Block::remove() {
  if (function_parent) {
    function_parent->body = nullptr;
    function_parent = nullptr;
  }
}

void format1(FormatStream &s, const Module *m) {
  format(s, FormatAddress(m), "  module\n");
  for (auto p : m->functions)
    format(s, p.second);
}

void format1(FormatStream &s, const Function *f) {
  format(s, FormatAddress(f), "    function ", f->return_type, " ", f->get_name(), "(");
  bool first = true;
  for (auto pt : f->parameter_types) {
    if (first)
      first = false;
    else
      format(s, ", ");
    format(s, pt);
  }
  format(s, ")\n");
}

void format1(FormatStream &s, const IR *x) {
  abort();
}

}
