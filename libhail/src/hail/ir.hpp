#ifndef HAIL_IR_HPP_INCLUDED
#define HAIL_IR_HPP_INCLUDED 1

#include <tuple>
#include <unordered_set>

#include <hail/allocators.hpp>
#include <hail/hash.hpp>
#include <hail/type.hpp>
#include <hail/value.hpp>

namespace hail {

class Module;
class Function;
class Block;
class Input;

class IRContextToken {
  friend class IRContext;
  IRContextToken() {}
};

class IRContext {
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

  void add_function(Function *f);
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

  void set_body(Block *new_body);

  void set_name(std::string new_name);
  const std::string &get_name() const { return name; }
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
    IF,
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
private:
  IR *parent;
  std::vector<IR *> children;
  std::unordered_set<std::tuple<IR *, int>> uses;
public:
  IR(Tag tag, IR *parent, size_t arity);
  IR(Tag tag, IR *parent, std::vector<IR *> children);
  virtual ~IR();

  void remove();

  const std::vector<IR *> &get_children() const { return children; }
  void set_children(std::vector<IR *> new_children);
  IR *get_child(size_t i) const { return children[i]; }
  void set_arity(size_t n) const;
  void set_child(size_t i, IR *x);

  IR *get_parent() const { return parent; }
  void set_parent(IR *new_parent);
};

class Block : public IR {
  friend class Function;

  Function *function_parent;
  std::vector<Input *> inputs;
  std::unordered_set<IR *> nodes;
public:
  static const Tag self_tag = IR::Tag::BLOCK;
  Block(IRContextToken, Function *function_parent, IR *parent, size_t arity, size_t input_arity);

  void remove();

  size_t get_input_arity() const;
  void set_input_arity(size_t n);

  const std::vector<Input *> &get_inputs() const { return inputs; }
  Input *get_input(size_t i) const;
  void set_input(size_t i, Input *) const;
};

class Input : public IR {
public:
  static const Tag self_tag = IR::Tag::INPUT;
  size_t index;
  Input(IR *parent, size_t index) : IR(self_tag, parent, 0), index(index) {}
};

class Literal : public IR {
public:
  static const Tag self_tag = IR::Tag::NA;
  Value value;
  Literal(IR *parent, Value value) : IR(self_tag, parent, 0), value(std::move(value)) {}

  void validate();
};

class NA : public IR {
public:
  static const Tag self_tag = IR::Tag::NA;
  const Type *type;
  NA(IR *parent, const Type *type) : IR(self_tag, parent, 0), type(type) {}
};

class IsNA : public IR {
public:
  static const Tag self_tag = IR::Tag::ISNA;
  IsNA(IR *value, IR *parent) : IR(self_tag, parent, {value}) {}
};

extern void format1(FormatStream &s, const Module *m);
extern void format1(FormatStream &s, const Function *f);
extern void format1(FormatStream &s, const IR *x);

}

#endif
