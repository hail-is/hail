
class IRContext {
  ArenaAllocator<HeapAllocator> arena;

public:
  IRContext(HeapAllocator &heap)
    : arena(heap) {}
  ~IRContext() {}

  Module *make_module();
  Function *make_function(Module *module,
			  std::string name,
			  std::vector<const Type *> parameter_types,
			  const Type *return_type);
};

class Module {
  std::map<std::string, Function *> functions;

public:
  Module();
  ~Module();

  void add_function(Function *f);
};

class Function {
  Module *module;
  std::string name;
  std::vector<const Type *> parameter_types;
  const Type *return_type;
  Block *body;

public:
  Function(Module *module,
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
  IR *parent;
  std::vector<IR *> children;
  std::unordered_set<std::tuple<IR *, int>> uses;
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

  virtual void validate() = 0;
};

class Block : public IR {
  std::vector<Input *> inputs;
  std::unordered_set<IR *> nodes;
public:
  static const Tag self_tag = IR::Tag::BLOCK;
  Block(size_t input_arity) : IR(self_tag), inputs(input_arity) {}

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
  Input(IR *parent, size_t index) : IR(self_tag, 0, parent), index(index) {}
};

class Literal : public IR {
public:
  static const Tag self_tag = IR::Tag::NA;
  Value value;
  NA(IR *parent, Value value) : IR(self_tag, parent, 0), value(std::move(value)) {}
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
  NA(IR *value, IR *parent) : IR(self_tag, parent, {value}), type(type) {}
};
