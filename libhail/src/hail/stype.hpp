enum class PrimitiveType {
  INT8,
  INT32,
  FLOAT32,
  FLOAT64,
  POINTER
};

// Make the scope keep unique pointers to the values and allocate when
// it goes out scope?

// You can't pre-evaluate, in every case the caller needs to assume
// the arguments into the desired location.  Yes.  That's the point.
// Or an EmitValue is a label to run, and two labels to consume.
// Maybe don't use block on if, but do use on e.g. StreamMap.

// Do we let LLVM stuff get on Type?  No.

// FIXME question, do we have two emit types, one with a missing byte
// and one with a pair of blocks?  The blocks can probably get cleaned
// up afterwards.

// FIXME question, what if we had a Python-like language users could
// write that we ran once per row or once per partition?

class SType {
public:
  const Type *const type;

  std::vector<PrimitiveType> constituent_types() const;
}

class SScalarType : public SType {
  virtual llvm::Value *get_llvm_value(const SScalarValue *value) const = 0;
};

class STupleType : public SType {
  virtual std::shared_ptr<EmitValue> get_element(const SValue *index) const = 0;
};

class SValue {
  virtual ~SValue();

public:
  const std::vector<llvm::Value *> &constituents() const = 0;

  std::shared_ptr<SValue> from_constituents(const std::vector<llvm::Value *> &constituents) const = 0;
};

class SScalarValue : public SValue {
public:
  virtual llvm::Value *get_llvm_value() const = 0;
}

class SStrValue : public SValue {
public:
  virtual llvm::Value *get_length() const = 0;

  virtual llvm::Value *get_data() const = 0;
};

class STupleValue : public SValue {
  virtual std::shared_ptr<EmitValue> get_element(const SValue *index) const = 0;
};

class SArrayValue : public SValue {
  virtual std::shared_ptr<EmitValue> get_element(const SValue *index) const = 0;
};

class EmitValue {
public:
  llvm::BasicBlock *present_block, *missing_block;
  std::shared_ptr<SValue> svalue;
};

