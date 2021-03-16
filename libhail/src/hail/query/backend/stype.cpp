#include <hail/query/backend/compile.hpp>
#include <hail/query/backend/stype.hpp>
#include <hail/query/backend/svalue.hpp>
#include <llvm/IR/Constants.h>

namespace hail {

SType::~SType() {}

void
SType::get_constituent_types(std::vector<PrimitiveType> &constituent_types) const {
  switch (tag) {
  case SType::Tag::BOOL:
    constituent_types.push_back(PrimitiveType::INT8);
    break;
  case SType::Tag::INT32:
    constituent_types.push_back(PrimitiveType::INT32);
    break;
  case SType::Tag::INT64:
    constituent_types.push_back(PrimitiveType::INT64);
    break;
  case SType::Tag::FLOAT32:
    constituent_types.push_back(PrimitiveType::FLOAT32);
    break;
  case SType::Tag::FLOAT64:
    constituent_types.push_back(PrimitiveType::FLOAT64);
    break;
  case SType::Tag::CANONICALTUPLE:
    constituent_types.push_back(PrimitiveType::POINTER);
    break;
  default:
    abort();
  }
}

SValue *
SType::from_llvm_values(const std::vector<llvm::Value *> &llvm_values, size_t i) const {
  switch (tag) {
  case SType::Tag::BOOL:
    return new SBoolValue(this, llvm_values[i]);
  case SType::Tag::INT32:
    return new SInt32Value(this, llvm_values[i]);
  case SType::Tag::INT64:
    return new SInt64Value(this, llvm_values[i]);
  case SType::Tag::FLOAT32:
    return new SFloat32Value(this, llvm_values[i]);
  case SType::Tag::FLOAT64:
    return new SFloat64Value(this, llvm_values[i]);
  case SType::Tag::CANONICALTUPLE:
    return new SCanonicalTupleValue(this, llvm_values[i]);
  default:
    abort();
  }
}

const SValue *
SType::load_from_address(CompileFunction &cf, llvm::Value *address) const {
  switch (tag) {
  case SType::Tag::BOOL:
    {
      auto v = cf.llvm_ir_builder.CreateLoad(address);
      return new SBoolValue(this, v);
    }
  case SType::Tag::INT32:
    {
      auto typed_address = cf.llvm_ir_builder.CreateBitCast(address,
							    llvm::PointerType::get(llvm::Type::getInt32Ty(cf.llvm_context), 0));
      auto v = cf.llvm_ir_builder.CreateLoad(typed_address);
      return new SInt32Value(this, v);
    }
  case SType::Tag::INT64:
    {
      auto typed_address = cf.llvm_ir_builder.CreateBitCast(address,
							    llvm::PointerType::get(llvm::Type::getInt64Ty(cf.llvm_context), 0));
      auto v = cf.llvm_ir_builder.CreateLoad(typed_address);
      return new SInt64Value(this, v);
    }
  case SType::Tag::FLOAT32:
    {
      auto typed_address = cf.llvm_ir_builder.CreateBitCast(address,
							    llvm::PointerType::get(llvm::Type::getFloatTy(cf.llvm_context), 0));
      auto v = cf.llvm_ir_builder.CreateLoad(typed_address);
      return new SFloat32Value(this, v);
    }
  case SType::Tag::FLOAT64:
    {
      auto typed_address = cf.llvm_ir_builder.CreateBitCast(address,
							    llvm::PointerType::get(llvm::Type::getDoubleTy(cf.llvm_context), 0));
      auto v = cf.llvm_ir_builder.CreateLoad(typed_address);
      return new SFloat64Value(this, v);
    }
  case SType::Tag::CANONICALTUPLE:
    {
      return new SCanonicalTupleValue(this, address);
    }
  default:
    abort();
  }
}

SBool::SBool(const Type *type)
  : SType(self_tag, type) {}

SInt32::SInt32(const Type *type)
  : SType(self_tag, type) {}

SInt64::SInt64(const Type *type)
  : SType(self_tag, type) {}

SFloat32::SFloat32(const Type *type)
  : SType(self_tag, type) {}

SFloat64::SFloat64(const Type *type)
  : SType(self_tag, type) {}

SCanonicalTuple::SCanonicalTuple(const Type *type, std::vector<const SType *> element_stypes, std::vector<size_t> element_offsets)
  : SType(self_tag, type),
    element_stypes(std::move(element_stypes)),
    element_offsets(std::move(element_offsets)) {}

void
EmitType::get_constituent_types(std::vector<PrimitiveType> &constituent_types) const {
  constituent_types.push_back(PrimitiveType::INT8);
  stype->get_constituent_types(constituent_types);
}

EmitValue
EmitType::from_llvm_values(const std::vector<llvm::Value *> &llvm_values, size_t i) const {
  llvm::Value *missing = llvm_values[i++];
  SValue *svalue = stype->from_llvm_values(llvm_values, i);
  return EmitValue(missing, svalue);
}

EmitValue
EmitType::make_na(CompileFunction &cf) const {
  auto m = llvm::ConstantInt::get(llvm::Type::getInt8Ty(cf.llvm_context), 1);

  std::vector<llvm::Value *> llvm_values;
  std::vector<PrimitiveType> constituent_types;
  stype->get_constituent_types(constituent_types);
  for (auto pt : constituent_types)
    llvm_values.push_back(llvm::UndefValue::get(cf.get_llvm_type(pt)));
  auto svalue = stype->from_llvm_values(llvm_values, 0);

  return EmitValue(m, svalue);
}

}
