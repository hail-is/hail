#ifndef HAIL_QUERY_BACKEND_SVALUE_HPP_INCLUDED
#define HAIL_QUERY_BACKEND_SVALUE_HPP_INCLUDED 1

#include <hail/query/backend/stype.hpp>
#include <vector>

namespace hail {

class CompileFunction;
class SType;

class SValue {
public:
  using BaseType = SValue;
  enum class Tag {
    BOOL,
    INT32,
    INT64,
    FLOAT32,
    FLOAT64
  };
  const Tag tag;
  const SType *const stype;
  SValue(Tag tag, const SType *stype) : tag(tag), stype(stype) {}

  virtual ~SValue();

  void get_constituent_values(std::vector<llvm::Value *> &llvm_values) const;
};

class SBoolValue : public SValue {
public:
  static const Tag self_tag = SValue::Tag::BOOL;
  llvm::Value *value;
  SBoolValue(const SType *stype, llvm::Value *value);
};

class SInt32Value : public SValue {
public:
  static const Tag self_tag = SValue::Tag::INT32;
  llvm::Value *value;
  SInt32Value(const SType *stype, llvm::Value *value);
};

class SInt64Value : public SValue {
public:
  static const Tag self_tag = SValue::Tag::INT64;
  llvm::Value *value;
  SInt64Value(const SType *stype, llvm::Value *value);
};

class SFloat32Value : public SValue {
public:
  static const Tag self_tag = SValue::Tag::FLOAT32;
  llvm::Value *value;
  SFloat32Value(const SType *stype, llvm::Value *value);
};

class SFloat64Value : public SValue {
public:
  static const Tag self_tag = SValue::Tag::FLOAT64;
  llvm::Value *value;
  SFloat64Value(const SType *stype, llvm::Value *value);
};

class EmitControlValue {
public:
  llvm::BasicBlock *present_block;
  llvm::BasicBlock *missing_block;
  const SValue *svalue;

  EmitControlValue(llvm::BasicBlock *present_block,
		   llvm::BasicBlock *missing_block,
		   const SValue *svalue)
    : present_block(present_block),
      missing_block(missing_block),
      svalue(svalue) {}
};

class EmitDataValue {
public:
  llvm::Value *missing;
  const SValue *svalue;

  EmitDataValue(llvm::Value *missing,
		const SValue *svalue)
    : missing(missing),
      svalue(svalue) {}

  void get_constituent_values(std::vector<llvm::Value *> &llvm_values) const;
};

class EmitValue {
  llvm::Value *missing;
  llvm::BasicBlock *present_block;
  llvm::BasicBlock *missing_block;
  const SValue *svalue;
public:
  EmitValue(llvm::Value *missing, const SValue *svalue);
  EmitValue(llvm::BasicBlock *present_block, llvm::BasicBlock *missing_block, const SValue *svalue);

  EmitControlValue as_control(CompileFunction &cf) const;
  EmitDataValue as_data(CompileFunction &cf) const;
};

}

#endif
