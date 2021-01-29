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
  size_t n = parameter_types.size();
  auto f = arena.make<Function>(IRContextToken(),
				module,
				std::move(name),
				// don't move, used below
				parameter_types,
				return_type);

  auto body = arena.make<Block>(IRContextToken(), *this, f, nullptr, 1, n);
  for (size_t i = 0; i < n; ++i)
    body->inputs[i] = arena.make<Input>(IRContextToken(), body, i);
  f->set_body(body);
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

void
Module::pretty_self(FormatStream &s) {
  format(s, FormatAddress(this), "  module\n");
  for (auto p : functions)
    p.second->pretty_self(s, 4);
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

void
Function::pretty_self(FormatStream &s, int indent) {
  format(s, FormatAddress(this), Indent(indent), "function ", return_type, " ", name, "(");
  bool first = true;
  for (auto pt : parameter_types) {
    if (first)
      first = false;
    else
      format(s, ", ");
    format(s, pt);
  }
  format(s, ")\n");
  pretty(s, get_body(), indent + 2);
  format(s, "\n");
}

IR::IR(Tag tag, Block *parent, size_t arity)
  : tag(tag), parent(parent), children(arity) {}

IR::IR(Tag tag, Block *parent, std::vector<IR *> children)
  : tag(tag), parent(parent), children(std::move(children)) {}

IR::~IR() {}

void
IR::add_use(IR *u, size_t i) {
  auto p = uses.insert({u, i});
  /* make sure the insert happened */
  assert(p.second);
}

void
IR::remove_use(IR *u, size_t i) {
  auto n = uses.erase({u, i});
  /* make sure the removal happened */
  assert(n == 1);
}

void
IR::set_child(size_t i, IR *x) {
  IR *c = children[i];
  if (c)
    c->remove_use(this, i);
  children[i] = x;
  x->add_use(this, i);
}

void
IR::pretty_self(FormatStream &s, int indent) {
  format(s, FormatAddress(this), Indent(indent), "???");
}

void
pretty(FormatStream &s, IR *x, int indent) {
  if (x) {
    x->pretty_self(s, indent);
    return;
  }

  format(s, FormatAddress(nullptr), Indent(indent), "null");
}

Block::Block(IRContextToken, IRContext &xc, Function *function_parent, Block *parent, std::vector<IR *> children, size_t input_arity)
  : IR(self_tag, parent, std::move(children)),
    xc(xc),
    function_parent(function_parent),
    inputs(input_arity) {}

Block::Block(IRContextToken, IRContext &xc, Function *function_parent, Block *parent, size_t arity, size_t input_arity)
  : IR(self_tag, parent, arity),
    xc(xc),
    function_parent(function_parent),
    inputs(input_arity) {}

void
Block::remove() {
  if (function_parent) {
    function_parent->body = nullptr;
    function_parent = nullptr;
  }
}

Block *
Block::make_block(size_t arity, size_t input_arity) {
  return xc.arena.make<Block>(IRContextToken(),
			      xc, nullptr, this, arity, input_arity);
}

Block *
Block::make_block(std::vector<IR *> children) {
  return xc.arena.make<Block>(IRContextToken(),
			      xc, nullptr, this, children, 0);
}

Input *
Block::make_input(size_t index) {
  return xc.arena.make<Input>(IRContextToken(), this, index);
}

Literal *
Block::make_literal(const Value &v) {
  return xc.arena.make<Literal>(IRContextToken(), this, v);
}

Mux *
Block::make_mux(IR *x, Block *tb, Block *fb) {
  return xc.arena.make<Mux>(IRContextToken(), this, x, tb, fb);
}

}
