#include <exception>

#include "hail/format.hpp"
#include "hail/query/ir.hpp"

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

Function *
Module::get_function(const std::string &name) {
  auto i = functions.find(name);
  assert(i != functions.end());
  return i->second;
}

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

const std::vector<std::string> IR::tag_name = {
  "block",
  "input",
  "literal",
  "na",
  "isna",
  "mux",
  "unary",
  "binary",
  "makearray",
  "arraylen",
  "arrayref",
  "maketuple",
  "gettupleelement",
  "tostream",
  "streammap",
  "streamflatmap",
  "streamfilter",
  "streamfold"
};

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
  format(s, FormatAddress(this), Indent(indent), "(", tag_name[static_cast<int>(tag)], " ...)");
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
  : IR(Tag::BLOCK, parent, std::move(children)),
    xc(xc),
    function_parent(function_parent),
    inputs(input_arity) {}

Block::Block(IRContextToken, IRContext &xc, Function *function_parent, Block *parent, size_t arity, size_t input_arity)
  : IR(Tag::BLOCK, parent, arity),
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

NA *
Block::make_na(const Type *t) {
  return xc.arena.make<NA>(IRContextToken(), this, t);
}

IsNA *
Block::make_is_na(IR *x) {
  return xc.arena.make<IsNA>(IRContextToken(), this, x);
}

Mux *
Block::make_mux(IR *x, IR *true_value, IR *false_value) {
  return xc.arena.make<Mux>(IRContextToken(), this, x, true_value, false_value);
}

MakeTuple *
Block::make_tuple(std::vector<IR *> elements) {
  return xc.arena.make<MakeTuple>(IRContextToken(), this, std::move(elements));
}

GetTupleElement *
Block::make_get_tuple_element(IR *t, int i) {
  return xc.arena.make<GetTupleElement>(IRContextToken(), this, t, i);
}

}
