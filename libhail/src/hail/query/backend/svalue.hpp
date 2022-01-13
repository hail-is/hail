#ifndef HAIL_QUERY_BACKEND_SVALUE_HPP_INCLUDED
#define HAIL_QUERY_BACKEND_SVALUE_HPP_INCLUDED 1

#include <vector>

#include "hail/query/backend/stype.hpp"

namespace hail {

class CompileFunction;
class SType;
class EmitValue;
class EmitDataValue;

class SValue {
public:
  using BaseType = SValue;
  enum class Tag {
    BOOL,
    INT32,
    INT64,
    FLOAT32,
    FLOAT64,
    CANONICALTUPLE,
    STACKTUPLE
  };
  const Tag tag;
  const SType *const stype;
  SValue(Tag tag, const SType *stype) : tag(tag), stype(stype) {}

  virtual ~SValue();

  void get_constituent_values(std::vector<llvm::Value *> &llvm_values) const;

  const SValue *cast_to(CompileFunction &cf, const SType *desired_type) const;
};

class SBoolValue : public SValue {
public:
  static bool is_instance_tag(Tag tag) { return tag == SValue::Tag::BOOL; }
  llvm::Value *value;
  SBoolValue(const SType *stype, llvm::Value *value);
};

class SInt32Value : public SValue {
public:
  static bool is_instance_tag(Tag tag) { return tag == SValue::Tag::INT32; }
  llvm::Value *value;
  SInt32Value(const SType *stype, llvm::Value *value);
};

class SInt64Value : public SValue {
public:
  static bool is_instance_tag(Tag tag) { return tag == SValue::Tag::INT64; }
  llvm::Value *value;
  SInt64Value(const SType *stype, llvm::Value *value);
};

class SFloat32Value : public SValue {
public:
  static bool is_instance_tag(Tag tag) { return tag == SValue::Tag::FLOAT32; }
  llvm::Value *value;
  SFloat32Value(const SType *stype, llvm::Value *value);
};

class SFloat64Value : public SValue {
public:
  static bool is_instance_tag(Tag tag) { return tag == SValue::Tag::FLOAT64; }
  llvm::Value *value;
  SFloat64Value(const SType *stype, llvm::Value *value);
};

class STupleValue : public SValue {
public:
  static bool is_instance_tag(Tag tag) {
    return (SValue::Tag::CANONICALTUPLE <= tag &&
	    tag <= SValue::Tag::STACKTUPLE);
  }
  STupleValue(Tag tag, const SType *stype);
  virtual EmitValue get_element(CompileFunction &cf, size_t i) const = 0;
};

class SCanonicalTupleValue : public STupleValue {
public:
  static bool is_instance_tag(Tag tag) { return tag == SValue::Tag::CANONICALTUPLE; }
  llvm::Value *address;
  SCanonicalTupleValue(const SType *stype, llvm::Value *address);
  EmitValue get_element(CompileFunction &cf, size_t i) const;
};

class SStackTupleValue : public STupleValue {
public:
  static bool is_instance_tag(Tag tag) { return tag == SValue::Tag::STACKTUPLE; }
  std::vector<EmitDataValue> element_emit_values;
  SStackTupleValue(const SType *stype, std::vector<EmitDataValue> element_emit_values);
  EmitValue get_element(CompileFunction &cf, size_t i) const;
};

class EmitControlValue {
public:
  llvm::BasicBlock *missing_block;
  const SValue *svalue;

  EmitControlValue(llvm::BasicBlock *missing_block,
		   const SValue *svalue)
    : missing_block(missing_block),
      svalue(svalue) {}

  EmitType get_type() const;
};

class EmitDataValue {
public:
  llvm::Value *missing;
  const SValue *svalue;

  EmitDataValue(llvm::Value *missing,
		const SValue *svalue)
    : missing(missing),
      svalue(svalue) {}

  EmitType get_type() const;

  void get_constituent_values(std::vector<llvm::Value *> &llvm_values) const;
};

class EmitValue {
  llvm::Value *missing;
  llvm::BasicBlock *missing_block;
  const SValue *svalue;
public:
  EmitValue(llvm::Value *missing, const SValue *svalue);
  EmitValue(llvm::BasicBlock *missing_block, const SValue *svalue);
  EmitValue(const EmitDataValue &data_value);
  EmitValue(const EmitControlValue &data_value);

  EmitType get_type() const;

  EmitControlValue as_control(CompileFunction &cf) const;
  EmitDataValue as_data(CompileFunction &cf) const;

  EmitValue cast_to(CompileFunction &cf, EmitType desired_type);
};

}

#endif
