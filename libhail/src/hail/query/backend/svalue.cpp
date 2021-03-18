#include <hail/query/backend/compile.hpp>
#include <hail/query/backend/svalue.hpp>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/IRBuilder.h>

namespace hail {

SValue::~SValue() {}

void
SValue::get_constituent_values(std::vector<llvm::Value *> &llvm_values) const {
  switch (tag) {
  case SValue::Tag::BOOL:
    llvm_values.push_back(cast<SBoolValue>(this)->value);
    break;
  case SValue::Tag::INT32:
    llvm_values.push_back(cast<SInt32Value>(this)->value);
    break;
  case SValue::Tag::INT64:
    llvm_values.push_back(cast<SInt64Value>(this)->value);
    break;
  case SValue::Tag::FLOAT32:
    llvm_values.push_back(cast<SFloat32Value>(this)->value);
    break;
  case SValue::Tag::FLOAT64:
    llvm_values.push_back(cast<SFloat64Value>(this)->value);
    break;
  default:
    abort();
  }
}

SBoolValue::SBoolValue(const SType *stype, llvm::Value *value)
  : SValue(self_tag, stype), value(value) {}


SInt32Value::SInt32Value(const SType *stype, llvm::Value *value)
  : SValue(self_tag, stype), value(value) {}

SInt64Value::SInt64Value(const SType *stype, llvm::Value *value)
  : SValue(self_tag, stype), value(value) {}

SFloat32Value::SFloat32Value(const SType *stype, llvm::Value *value)
  : SValue(self_tag, stype), value(value) {}

SFloat64Value::SFloat64Value(const SType *stype, llvm::Value *value)
  : SValue(self_tag, stype), value(value) {}

void
EmitDataValue::get_constituent_values(std::vector<llvm::Value *> &llvm_values) const {
  llvm_values.push_back(missing);
  svalue->get_constituent_values(llvm_values);
}

EmitValue::EmitValue(llvm::Value *missing, const SValue *svalue)
  : missing(missing),
    present_block(nullptr),
    missing_block(nullptr),
    svalue(svalue) {}

EmitValue::EmitValue(llvm::BasicBlock *present_block, llvm::BasicBlock *missing_block, const SValue *svalue)
  : missing(nullptr),
    present_block(present_block),
    missing_block(missing_block),
    svalue(svalue) {}

EmitControlValue
EmitValue::as_control(CompileFunction &cf) const {
  if (!missing)
    return EmitControlValue(present_block, missing_block, svalue);

  auto present_bb = llvm::BasicBlock::Create(cf.llvm_context, "as_control_present", cf.llvm_function);
  auto missing_bb = llvm::BasicBlock::Create(cf.llvm_context, "as_control_missing", cf.llvm_function);

  auto i8zero = llvm::ConstantInt::get(llvm::Type::getInt8Ty(cf.llvm_context), 0);
  auto cond = cf.llvm_ir_builder.CreateICmpNE(missing, i8zero);
  cf.llvm_ir_builder.CreateCondBr(cond, missing_bb, present_bb);

  return EmitControlValue(present_bb, missing_bb, svalue);
}

EmitDataValue
EmitValue::as_data(CompileFunction &cf) const {
  if (missing)
    return EmitDataValue(missing, svalue);

  llvm::AllocaInst *l = cf.make_entry_alloca(llvm::Type::getInt8Ty(cf.llvm_context));

  auto merge_bb = llvm::BasicBlock::Create(cf.llvm_context, "as_data", cf.llvm_function);
  cf.llvm_ir_builder.SetInsertPoint(missing_block);
  cf.llvm_ir_builder.CreateStore(llvm::ConstantInt::get(llvm::Type::getInt8Ty(cf.llvm_context), 1), l);
  cf.llvm_ir_builder.CreateBr(merge_bb);

  cf.llvm_ir_builder.SetInsertPoint(present_block);
  cf.llvm_ir_builder.CreateStore(llvm::ConstantInt::get(llvm::Type::getInt8Ty(cf.llvm_context), 0), l);
  cf.llvm_ir_builder.CreateBr(merge_bb);

  cf.llvm_ir_builder.SetInsertPoint(merge_bb);
  llvm::Value *m = cf.llvm_ir_builder.CreateLoad(l);

  return EmitDataValue(m, svalue);
}

}
