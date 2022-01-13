#ifndef HAIL_IR_HPP_INCLUDED
#define HAIL_IR_HPP_INCLUDED 1

#include <tuple>
#include <unordered_set>
#include <string>

#include "hail/allocators.hpp"
#include "hail/hash.hpp"
#include "hail/type.hpp"
#include "hail/value.hpp"

namespace hail {

class Module;
class Function;
class Block;
class Input;
class Literal;
class NA;
class IsNA;
class Mux;
class MakeArray;
class ArrayRef;
class ArrayLen;
class MakeTuple;
class GetTupleElement;

class IRContextToken {
  friend class IRContext;
  friend class Block;
  IRContextToken() {}
};

class IRContext {
  friend class Block;

  ArenaAllocator arena;

public:
  IRContext(HeapAllocator &heap);
  ~IRContext();

  Module *make_module();
  Function *make_function(Module *module,
			  std::string name,
			  std::vector<const Type *> parameter_types,
			  const Type *return_type);
};

class Module {
  friend class Function;
  friend class IR;
  friend void format1(FormatStream &s, const Module *m);

  std::map<std::string, Function *> functions;

public:
  Module(IRContextToken);
  ~Module();

  Function *get_function(const std::string &name);
  const std::map<std::string, Function *> &get_functions() const { return functions; }
  void add_function(Function *f);

  void pretty_self(FormatStream &s);
};

class Function {
  friend class Module;
  friend class IR;
  friend class Block;

  Module *module;
  std::string name;
public:
  std::vector<const Type *> parameter_types;
  const Type *return_type;
private:
  Block *body;

public:
  Function(IRContextToken,
	   Module *module,
	   std::string name,
	   std::vector<const Type *> parameter_types,
	   const Type *return_type);
  ~Function();

  void remove();

  Block *get_body() const { return body; }
  void set_body(Block *new_body);

  void set_name(std::string new_name);
  const std::string &get_name() const { return name; }

  void pretty_self(FormatStream &s, int indent = 0);
};

class IR {
  friend class Module;
  friend class Function;

public:
  using BaseType = IR;
  enum class Tag {
    BLOCK,
    INPUT,
    LITERAL,
    NA,
    ISNA,
    MUX,
    UNARY,
    BINARY,
    MAKEARRAY,
    ARRAYLEN,
    ARRAYREF,
    MAKETUPLE,
    GETTUPLEELEMENT,
    TOSTREAM,
    STREAMMAP,
    STREAMFLATMAP,
    STREAMFILTER,
    STREAMFOLD
  };
  const Tag tag;
  static const std::vector<std::string> tag_name;
private:
  Block *parent;
  std::vector<IR *> children;
  std::unordered_set<std::tuple<IR *, size_t>> uses;
public:
  IR(Tag tag, Block *parent, size_t arity);
  IR(Tag tag, Block *parent, std::vector<IR *> children);
  virtual ~IR();

  void remove_use(IR *u, size_t i);
  void add_use(IR *u, size_t i);

  const std::vector<IR *> &get_children() const { return children; }
  void set_children(std::vector<IR *> new_children);
  IR *get_child(size_t i) const { return children[i]; }
  void set_arity(size_t n) const;
  void set_child(size_t i, IR *x);

  Block *get_parent() const { return parent; }
  void set_parent(Block *new_parent);

  virtual void pretty_self(FormatStream &s, int indent);
  void pretty_self(FormatStream &s);

  template<typename F> auto dispatch(F f);
};

extern void pretty(FormatStream &s, IR *x, int indent);

class Block : public IR {
  friend class Function;

  IRContext &xc;
  Function *function_parent;
  std::unordered_set<IR *> nodes;
public:
  static bool is_instance_tag(Tag tag) { return tag == Tag::BLOCK; }
  std::vector<Input *> inputs;
  Block(IRContextToken, IRContext &xc, Function *function_parent, Block *parent, std::vector<IR *> children, size_t input_arity);
  Block(IRContextToken, IRContext &xc, Function *function_parent, Block *parent, size_t arity, size_t input_arity);

  void remove();

  Function *get_function_parent() const { return function_parent; }
  size_t get_input_arity() const;
  void set_input_arity(size_t n);

  const std::vector<Input *> &get_inputs() const { return inputs; }

  /* Builder methods for building IR. */
  Block *make_block(size_t arity, size_t input_arity);
  Block *make_block(std::vector<IR *> children);
  Input *make_input(size_t index);
  Literal *make_literal(const Value &v);
  NA *make_na(const Type *type);
  IsNA *make_is_na(IR *x);
  Mux *make_mux(IR *x, IR *true_value, IR *false_value);
  MakeArray *make_make_array(std::vector<IR *> elements);
  MakeArray *make_make_array(const Type *element_type, std::vector<IR *> children);
  ArrayLen *make_array_len(IR *a, IR *x);
  MakeTuple *make_tuple(std::vector<IR *> elements);
  GetTupleElement *make_get_tuple_element(IR *t, int i);
};

class Input : public IR {
public:
  static bool is_instance_tag(Tag tag) { return tag == Tag::INPUT; }
  size_t index;
  Input(IRContextToken, Block *parent, size_t index) : IR(Tag::INPUT, parent, 0), index(index) {}
};

class Literal : public IR {
public:
  static bool is_instance_tag(Tag tag) { return tag == Tag::LITERAL; }
  Value value;
  Literal(IRContextToken, Block *parent, Value value) : IR(Tag::LITERAL, parent, 0), value(std::move(value)) {}
};

class NA : public IR {
public:
  static bool is_instance_tag(Tag tag) { return tag == Tag::NA; }
  const Type *type;
  NA(IRContextToken, Block *parent, const Type *type) : IR(Tag::NA, parent, 0), type(type) {}
};

class Mux : public IR {
public:
  static bool is_instance_tag(Tag tag) { return tag == Tag::MUX; }
  Mux(IRContextToken, Block *parent, IR *condition, IR *true_value, IR *false_value)
    : IR(Tag::MUX, parent, {condition, true_value, false_value}) {}
};

class IsNA : public IR {
public:
  static bool is_instance_tag(Tag tag) { return tag == Tag::ISNA; }
  IsNA(IRContextToken, Block *parent, IR *x) : IR(Tag::ISNA, parent, {x}) {}
};

class MakeArray : public IR {
public:
  static bool is_instance_tag(Tag tag) { return tag == Tag::MAKEARRAY; }
  MakeArray(IRContextToken, Block *parent, std::vector<IR *> elements) : IR(Tag::MAKEARRAY, parent, std::move(elements)) {}
};

class ArrayLen : public IR {
public:
  static bool is_instance_tag(Tag tag) { return tag == Tag::ARRAYLEN; }
  ArrayLen(IRContextToken, Block *parent, IR *a) : IR(Tag::ARRAYLEN, parent, {a}) {}
};

class ArrayRef : public IR {
public:
  static bool is_instance_tag(Tag tag) { return tag == Tag::ARRAYREF; }
  ArrayRef(IRContextToken, Block *parent, IR *a, IR *x) : IR(Tag::ARRAYREF, parent, {a, x}) {}
};

class MakeTuple : public IR {
public:
  static bool is_instance_tag(Tag tag) { return tag == Tag::MAKETUPLE; }
  MakeTuple(IRContextToken, Block *parent, std::vector<IR *> elements) : IR(Tag::MAKETUPLE, parent, std::move(elements)) {}
};

class GetTupleElement : public IR {
public:
  static bool is_instance_tag(Tag tag) { return tag == Tag::GETTUPLEELEMENT; }
  size_t index;
  GetTupleElement(IRContextToken, Block *parent, IR *t, size_t index) : IR(Tag::GETTUPLEELEMENT, parent, {t}), index(index) {}
};

template<typename F> auto
IR::dispatch(F f) {
  switch(tag) {
  case IR::Tag::BLOCK: return f(cast<Block>(this));
  case IR::Tag::INPUT: return f(cast<Input>(this));
  case IR::Tag::LITERAL: return f(cast<Literal>(this));
  case IR::Tag::NA: return f(cast<NA>(this));
  case IR::Tag::ISNA: return f(cast<IsNA>(this));
  case IR::Tag::MAKETUPLE: return f(cast<MakeTuple>(this));
  case IR::Tag::GETTUPLEELEMENT: return f(cast<GetTupleElement>(this));
  default:
    abort();
  }
}

}

#endif
